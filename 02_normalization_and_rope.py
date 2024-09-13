import torch
import torch.nn as nn

# Device assignment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the RMSNorm normalization layer
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim).to(device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

# Rotary Positional Encoding (RoPE) precomputation
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs).to(device)
    freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out

# Test RMSNorm (remove comments to test)
x = torch.randn((10, 256, 512), device=device)
rms_norm = RMSNorm(dim=512)
x_norm = rms_norm(x)
print(f"Shape of x: {x.shape}")
print(f"Shape of normalized x: {x_norm.shape}")

"""
RMSNorm Result:-
Shape of x: torch.Size([10, 256, 512])
Shape of normalized x: torch.Size([10, 256, 512])

"""

# Test RoPE (remove comments to test)
head_dim = 64
seq_len = 256
freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=seq_len)
print(f"RoPE frequency shape: {freqs_cis.shape}")

"""
RoPE Result:-
RoPE frequency shape: torch.Size([256, 32])

"""

