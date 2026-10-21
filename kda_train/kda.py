"""KDA attention wrapper with FLA's CUDA kernels and a readable Torch fallback."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F


def _fla_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise RuntimeError(
            "The CUDA KDA backend needs fla-core. Install this project with: "
            "pip install -e '.[cuda]'"
        ) from exc

    out, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate_logits,
        beta=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    return out


def recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Readable differentiable KDA scan. This is intentionally not a fast path.

    Inputs q/k/v have shape [batch, time, heads, head_dim]. The scan keeps its
    recurrent state in float32 for numerical stability.
    """
    batch, time, heads, head_dim = q.shape
    value_dim = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    qf = F.normalize(qf, p=2.0, dim=-1, eps=1e-6) * (head_dim**-0.5)
    kf = F.normalize(kf, p=2.0, dim=-1, eps=1e-6)
    gf = gate_logits.float() + dt_bias.float().view(1, 1, heads, head_dim)
    beta = torch.sigmoid(beta_logits.float()).unsqueeze(-1)
    decay_rate = torch.exp(a_log.float()).view(1, heads, 1)

    state = torch.zeros(
        batch, heads, head_dim, value_dim, dtype=torch.float32, device=q.device
    )
    outputs: list[torch.Tensor] = []
    for t in range(time):
        alpha = torch.exp(-decay_rate * F.softplus(gf[:, t]))
        state = state * alpha.unsqueeze(-1)
        kt = kf[:, t]
        qt = qf[:, t]
        vt = vf[:, t]
        predicted = torch.einsum("bhk,bhkv->bhv", kt, state)
        write = beta[:, t] * (vt - predicted)
        state = state + torch.einsum("bhk,bhv->bhkv", kt, write)
        outputs.append(torch.einsum("bhk,bhkv->bhv", qt, state))

    return torch.stack(outputs, dim=1).to(dtype=q.dtype)


class KDAAttention(nn.Module):
    """Compact KDA block using channel-wise decay gates.

    CUDA training uses the chunkwise Triton implementation in fla-core. The
    Torch scan exists for CPU execution, inspection, and small experiments.
    """

    def __init__(self, dim: int, num_heads: int, backend: str = "auto") -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if backend not in {"auto", "fla", "reference"}:
            raise ValueError("backend must be one of: auto, fla, reference")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.backend = backend

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=True)
        self.beta_proj = nn.Linear(dim, num_heads, bias=True)
        self.z_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.o_norm = RMSNorm(self.head_dim)

        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(num_heads * self.head_dim, dtype=torch.float32)
        )
        with torch.no_grad():
            self.A_log.copy_(torch.empty(num_heads).uniform_(1.0, 16.0).log())
            dt = torch.exp(
                torch.empty(num_heads * self.head_dim).uniform_(
                    torch.log(torch.tensor(0.001)).item(),
                    torch.log(torch.tensor(0.1)).item(),
                )
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _use_fla(self, x: torch.Tensor) -> bool:
        if self.backend == "reference":
            return False
        if self.backend == "fla":
            if not x.is_cuda:
                raise RuntimeError("backend='fla' requires a CUDA tensor")
            return True
        if x.is_cuda:
            try:
                import fla.ops.kda  # noqa: F401
                return True
            except ImportError:
                warnings.warn(
                    "CUDA was detected but fla-core is not installed; using the "
                    "slow differentiable PyTorch KDA scan. Install with "
                    "pip install -e '.[cuda]' for the CUDA kernel.",
                    stacklevel=2,
                )
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = F.silu(self.q_proj(x)).view(batch, time, self.num_heads, self.head_dim)
        k = F.silu(self.k_proj(x)).view(batch, time, self.num_heads, self.head_dim)
        v = F.silu(self.v_proj(x)).view(batch, time, self.num_heads, self.head_dim)
        gate_logits = self.g_proj(x).view(
            batch, time, self.num_heads, self.head_dim
        )
        beta_logits = self.beta_proj(x)

        if self._use_fla(x):
            y = _fla_chunk_kda(
                q, k, v, gate_logits, beta_logits, self.A_log, self.dt_bias
            )
        else:
            y = recurrent_kda_reference(
                q, k, v, gate_logits, beta_logits, self.A_log, self.dt_bias
            )

        z = self.z_proj(x).view(batch, time, self.num_heads, self.head_dim)
        y = self.o_norm(y) * torch.sigmoid(z)
        return self.o_proj(y.reshape(batch, time, self.dim))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).to(dtype=x.dtype)

