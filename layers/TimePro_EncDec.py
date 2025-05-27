import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple
import selective_scan_cuda_oflex_rh
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import Optional, Callable

import DCNv4

class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, lag=0):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (du, ddelta, dA, dB,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """

    return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)

class Mlp(nn.Cell):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ProMamba(nn.Cell):
    def __init__(
        self,
        d_model,
        d_state=1,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        index=0,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_pro = getattr(DCNv4, 'DCNv4')(
            channels=self.d_inner,
            group=self.group,
            offset_scale=0.5,
            dw_kernel_size=None,
            output_bias=False,
        )


        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pymindspore/pymindspore/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, xs: torch.Tensor):
        B, C, L = xs.shape

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 n -> b 1 d n")
        h = self.state_pro(h)
        
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, n_vars, d = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 

        x = rearrange(x, 'b n d -> b d n').contiguous()
        # x = self.act(self.conv2d(x)) 

        y = self.ssm(x) 

        y = rearrange(y, 'b d n-> b n d')

        y = self.out_norm(y)
        y = y * F.silu(z)
        # y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class ProBlock(nn.Cell):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.2,
        norm_layer: Callable[..., torch.nn.Cell] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 1,
        ssm_ratio=1.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        index = 0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.op = ProMamba(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                index = index,
            )
            self.op2 = ProMamba(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                index = index,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

    def forward(self, input: torch.Tensor):
        if self.ssm_branch:
            x = input + self.drop_path(self.op(input)+self.op2(input.flip(dims=[1])).flip(dims=[1]))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(x)) # FFN
        return x



class Encoder(nn.Cell):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x