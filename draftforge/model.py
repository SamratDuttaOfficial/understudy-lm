"""Draft model: small Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU, tied
embeddings) on the target's tokenizer, plus optional Medusa lookahead heads."""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelArgs:
    vocab_size: int = 248320
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_mult: float = 2.75
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0
    n_extra_heads: int = 0          # Medusa heads for t+2 .. t+1+n

    @classmethod
    def from_config(cls, cfg, vocab_size: int, n_extra_heads: int = 0) -> "ModelArgs":
        return cls(vocab_size=vocab_size, d_model=cfg.d_model,
                   n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                   n_kv_heads=cfg.n_kv_heads, ffn_mult=cfg.ffn_mult,
                   max_seq_len=cfg.seq_len, rope_theta=cfg.rope_theta,
                   tie_embeddings=cfg.tie_embeddings, dropout=cfg.dropout,
                   n_extra_heads=n_extra_heads)


# Full logits for an 8x1024 batch at a 248K vocab are 7.6 GiB, so the loss runs
# a chunk of rows at a time and forms each chunk's gradient immediately.
LOSS_CHUNK_BYTES = 768 << 20


def loss_chunk_rows(vocab: int) -> int:
    # ~6 bytes per logit: fp32 plus one half-precision copy
    n = max(64, LOSS_CHUNK_BYTES // (vocab * 6))
    return 1 << (n.bit_length() - 1)


def _ce_chunks(h, weight, targets, row_w, chunk: int, cdt, need_grad: bool):
    """sum_i row_w[i] * CE(h[i] @ weight.T, targets[i]), plus its gradients."""
    w_c = weight.to(cdt)
    loss = torch.zeros((), device=h.device, dtype=torch.float32)
    gh = torch.empty_like(h, dtype=torch.float32) if need_grad else None
    gw = torch.zeros_like(weight, dtype=torch.float32) if need_grad else None
    for s in range(0, h.size(0), chunk):
        hc = h[s: s + chunk].to(cdt)
        t = targets[s: s + chunk, None]
        rw = row_w[s: s + chunk, None]
        logits = hc @ w_c.t()                                  # (C, V)
        # memory-bound: manual logsumexp, gradient reuses the logits buffer
        tgt = logits.gather(1, t).float()
        mx = logits.amax(-1, keepdim=True).float()
        p = torch.sub(logits, mx).exp_()                       # fp32
        se = p.sum(-1, keepdim=True)
        lse = mx + se.log()
        loss += ((lse - tgt) * rw).sum()
        if not need_grad:
            continue
        # softmax - onehot; target entry from fp32 so small gradients survive
        g = torch.mul(p, rw / se, out=logits)
        del p
        g.scatter_(1, t, ((torch.exp(tgt - lse) - 1) * rw).to(cdt))
        gh[s: s + chunk] = g @ w_c
        _addmm_into(gw, g.t(), hc)
    return loss, gh, gw


def _addmm_into(acc, a, b) -> None:
    """acc += a @ b, accumulated into fp32 without a temporary."""
    if a.dtype == acc.dtype:
        acc.addmm_(a, b)
        return
    try:
        torch.addmm(acc, a, b, out_dtype=acc.dtype, out=acc)
    except TypeError:                   # torch without out_dtype
        acc += a @ b


class _LinearCE(torch.autograd.Function):
    """Chunked loss; gradients are computed in forward and scaled in backward."""

    @staticmethod
    def forward(ctx, h, weight, targets, row_w, chunk, cdt):
        with torch.autocast(h.device.type, enabled=False):
            loss, ctx.gh, ctx.gw = _ce_chunks(h, weight, targets, row_w,
                                              chunk, cdt, True)
        ctx.dtypes = (h.dtype, weight.dtype)
        return loss

    @staticmethod
    def backward(ctx, go):
        gh, gw = ctx.gh.mul_(go), ctx.gw.mul_(go)
        ctx.gh = ctx.gw = None
        return (gh.to(ctx.dtypes[0]), gw.to(ctx.dtypes[1]),
                None, None, None, None)


def linear_cross_entropy(hs, weight, targets, scales, ignore_index: int = -100,
                         chunk: int | None = None):
    """sum_k scales[k] * mean CE(hs[k] @ weight.T, targets[k]), chunked."""
    d = weight.size(1)
    h = torch.cat([x.reshape(-1, d) for x in hs]) if len(hs) > 1 else hs[0].reshape(-1, d)
    ts, ws = [], []
    for y, s in zip(targets, scales):
        y = y.reshape(-1)
        valid = (y != ignore_index).float()
        ws.append(valid * (s / valid.sum().clamp(min=1)))
        ts.append(y.clamp(min=0))
    t = torch.cat(ts) if len(ts) > 1 else ts[0]
    row_w = torch.cat(ws) if len(ws) > 1 else ws[0]

    dev = h.device.type
    cdt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else h.dtype
    chunk = chunk or loss_chunk_rows(weight.size(0))
    if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
        return _LinearCE.apply(h, weight, t, row_w, chunk, cdt)
    with torch.autocast(dev, enabled=False):
        return _ce_chunks(h, weight, t, row_w, chunk, cdt, False)[0]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def build_rope(head_dim: int, max_seq: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        assert a.d_model % a.n_heads == 0
        assert a.n_heads % a.n_kv_heads == 0
        self.n_heads, self.n_kv_heads = a.n_heads, a.n_kv_heads
        self.hd = a.d_model // a.n_heads
        self.rep = a.n_heads // a.n_kv_heads
        self.wq = nn.Linear(a.d_model, a.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(a.d_model, a.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(a.d_model, a.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(a.n_heads * self.hd, a.d_model, bias=False)
        self.dropout = a.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        o = o.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(o)


class FeedForward(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        hidden = int(a.ffn_mult * a.d_model)
        hidden = 64 * ((hidden + 63) // 64)      # keep tensor cores happy
        self.w1 = nn.Linear(a.d_model, hidden, bias=False)
        self.w3 = nn.Linear(a.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, a.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.n1 = RMSNorm(a.d_model)
        self.attn = Attention(a)
        self.n2 = RMSNorm(a.d_model)
        self.ff = FeedForward(a)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ff(self.n2(x))


class MedusaHead(nn.Module):
    """One residual MLP block; its output goes through the shared lm_head."""

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.fc.weight)      # starts as identity

    def forward(self, h):
        return h + F.silu(self.fc(h))


class DraftLM(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.args = a
        self.tok = nn.Embedding(a.vocab_size, a.d_model)
        self.blocks = nn.ModuleList([Block(a) for _ in range(a.n_layers)])
        self.norm = RMSNorm(a.d_model)
        self.lm_head = nn.Linear(a.d_model, a.vocab_size, bias=False)
        if a.tie_embeddings:
            self.lm_head.weight = self.tok.weight
        self.extra_heads = nn.ModuleList(
            [MedusaHead(a.d_model) for _ in range(a.n_extra_heads)])

        cos, sin = build_rope(a.d_model // a.n_heads, a.max_seq_len, a.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("w2.weight") or n.endswith("wo.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * a.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def add_extra_heads(self, n: int) -> None:
        if n <= len(self.extra_heads):
            return
        dev = self.tok.weight.device
        for _ in range(n - len(self.extra_heads)):
            self.extra_heads.append(MedusaHead(self.args.d_model).to(dev))
        self.args.n_extra_heads = n

    def backbone(self, idx):
        B, T = idx.shape
        assert T <= self.args.max_seq_len, "sequence longer than max_seq_len"
        x = self.tok(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        # Recompute blocks in backward: ~1.4 GB saved for ~5% speed.
        ckpt = self.training and torch.is_grad_enabled()
        for b in self.blocks:
            x = checkpoint(b, x, cos, sin, use_reentrant=False) if ckpt else b(x, cos, sin)
        return self.norm(x)

    def forward(self, idx, targets=None, extra_targets=None, head_weight=0.4):
        """With targets, returns (None, loss, h); full logits are never built."""
        h = self.backbone(idx)
        if targets is None:
            return self.lm_head(h), None, h

        hs, ts, ws = [h], [targets], [1.0]
        if extra_targets is not None:
            for i, head in enumerate(self.extra_heads):
                if i >= extra_targets.size(1):
                    break
                hs.append(head(h))
                ts.append(extra_targets[:, i])
                ws.append(head_weight)
        loss = linear_cross_entropy(hs, self.lm_head.weight, ts, ws)
        return None, loss, h

    @torch.no_grad()
    def draft(self, idx, k: int, temperature: float = 0.0):
        """Propose k tokens: one pass with heads, else k sequential passes."""
        if len(self.extra_heads) >= k - 1 and k > 1:
            h = self.backbone(idx)[:, -1:]
            outs = [self.lm_head(h)]
            for i in range(k - 1):
                outs.append(self.lm_head(self.extra_heads[i](h)))
            logits = torch.cat(outs, dim=1)                    # (B, k, V)
            return self._pick(logits, temperature)
        seq = idx
        picks = []
        for _ in range(k):
            logits = self.lm_head(self.backbone(seq)[:, -1:])
            nxt = self._pick(logits, temperature)
            picks.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
            if seq.size(1) >= self.args.max_seq_len:
                break
        return torch.cat(picks, dim=1)

    @staticmethod
    def _pick(logits, temperature: float):
        if temperature <= 0:
            return logits.argmax(-1)
        p = F.softmax(logits / temperature, dim=-1)
        B, K, V = p.shape
        s = torch.multinomial(p.reshape(-1, V), 1).reshape(B, K)
        return s

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok.weight.numel()
            if not self.args.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def param_report(self) -> dict:
        emb = self.tok.weight.numel()
        head = 0 if self.args.tie_embeddings else self.lm_head.weight.numel()
        extra = sum(p.numel() for p in self.extra_heads.parameters())
        total = self.n_params()
        return {"total": total, "embedding": emb, "lm_head": head,
                "extra_heads": extra, "trunk": total - emb - head - extra}
