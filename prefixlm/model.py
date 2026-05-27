from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        batch: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        prefix_len: int,
        rope_theta: float = 10000.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0 <= prefix_len <= seq_len:
            raise ValueError("prefix_len must satisfy 0 <= prefix_len <= seq_len")

        self.batch = batch
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.prefix_len = prefix_len
        self.rope_theta = rope_theta

        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        rope_cos, rope_sin = self._build_rope_cache(
            seq_len, self.head_dim, rope_theta, device
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.block_mask = create_block_mask(
            self._mask_mod(prefix_len),
            B=batch,
            H=num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

    @staticmethod
    def _mask_mod(prefix_len: int):
        def mask_mod(b, h, q_idx, kv_idx):
            q_is_prefix = q_idx < prefix_len
            kv_is_prefix = kv_idx < prefix_len

            prefix_to_prefix = q_is_prefix & kv_is_prefix
            mixture_to_prefix = (~q_is_prefix) & kv_is_prefix
            mixture_to_causal_mixture = (
                (~q_is_prefix) & (~kv_is_prefix) & (kv_idx <= q_idx)
            )

            return prefix_to_prefix | mixture_to_prefix | mixture_to_causal_mixture

        return mask_mod

    @staticmethod
    def _score_mod(prefix_len: int):
        def score_mod(score, b, h, q_idx, kv_idx):
            q_is_prefix = q_idx < prefix_len
            kv_is_prefix = kv_idx < prefix_len

            prefix_to_prefix = q_is_prefix & kv_is_prefix
            mixture_to_prefix = (~q_is_prefix) & kv_is_prefix
            mixture_to_causal_mixture = (
                (~q_is_prefix) & (~kv_is_prefix) & (kv_idx <= q_idx)
            )
            mask = prefix_to_prefix | mixture_to_prefix | mixture_to_causal_mixture
            score = score.float()
            return torch.where(mask, score, torch.full_like(score, -float("inf")))

        return score_mod

    @staticmethod
    def _build_rope_cache(
        seq_len: int,
        head_dim: int,
        rope_theta: float,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        channel_indices = torch.arange(
            0, head_dim, 2, device=device, dtype=torch.float32
        )
        inv_freq = 1.0 / (rope_theta ** (channel_indices / head_dim))
        freqs = torch.outer(positions, inv_freq)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.rope_cos.to(dtype=x.dtype)
        sin = self.rope_sin.to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        if batch != self.batch or seq_len != self.seq_len:
            raise ValueError(
                f"Expected input shape ({self.batch}, {self.seq_len}, {self.embed_dim}), "
                f"got ({batch}, {seq_len}, {channels})"
            )
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected input embed_dim {self.embed_dim}, got {channels}"
            )

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        out = flex_attention(
            q,
            k,
            v,
            score_mod=self._score_mod(self.prefix_len),
            block_mask=self.block_mask,
        )

        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


def torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_model(
    batch: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    prefix_len: int,
    dtype: str,
    seed: int,
    rope_theta: float = 10000.0,
    device: torch.device | str = "cpu",
) -> MultiHeadAttention:
    torch.manual_seed(seed)
    model = MultiHeadAttention(
        batch=batch,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        prefix_len=prefix_len,
        rope_theta=rope_theta,
        device=device,
    ).eval()
    model.to(device=device, dtype=torch_dtype(dtype))
    model.requires_grad_(False)
    return model


if __name__ == "__main__":
    batch = 66
    heads = 4
    head_dim = 32
    num_prompts = 2
    num_frames = 376

    has_sos = True  # True if sequence has <SOS> token

    seq_len = num_prompts + int(has_sos) + num_frames
    prefix_len = num_prompts + int(has_sos)
    embed_dim = heads * head_dim
    device = "cuda"
    dtype = "float16"

    x = torch.rand(
        batch,
        seq_len,
        embed_dim,
        dtype=torch_dtype(dtype),
        device=device,
        requires_grad=True,
    )
    attn = build_model(
        batch=batch,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=heads,
        prefix_len=prefix_len,
        dtype=dtype,
        seed=0,
        device=device,
    )

    out = attn(x)
    print(out.shape)  # Expected: (batch, seq_len, embed_dim)
