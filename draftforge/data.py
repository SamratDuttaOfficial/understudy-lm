"""Flat uint32 token files on disk and batch sampling."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .paths import P
from .util import LOG, read_json, write_json

DTYPE = np.uint32
ITEM = np.dtype(DTYPE).itemsize


def bin_path(split: str) -> Path:
    return P.tokens / (split + ".bin")


def meta_path() -> Path:
    return P.tokens / "meta.json"


def append_tokens(split: str, arr: np.ndarray) -> int:
    p = bin_path(split)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr, dtype=DTYPE)
    with open(p, "ab") as f:
        f.write(a.tobytes())
        f.flush()
    return a.size


def token_count(split: str) -> int:
    p = bin_path(split)
    return p.stat().st_size // ITEM if p.exists() else 0


def truncate_to(split: str, n_tokens: int) -> None:
    """Roll a .bin back to its last committed length."""
    p = bin_path(split)
    if p.exists() and p.stat().st_size != n_tokens * ITEM:
        LOG.warning("truncating %s from %d to %d tokens (crash recovery)",
                    p.name, p.stat().st_size // ITEM, n_tokens)
        with open(p, "r+b") as f:
            f.truncate(n_tokens * ITEM)


def save_meta(meta: dict) -> None:
    write_json(meta_path(), meta)


def load_meta() -> dict:
    return read_json(meta_path(), {}) or {}


def chat_markers(cfg):
    """Token ids that frame an assistant answer, or None without a tokenizer."""
    from .paths import pin_caches
    pin_caches()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.target_hf, cache_dir=str(P.hf))
    except Exception as e:
        LOG.warning("tokenizer unavailable (%s) - cannot tell answers from "
                    "prompts", e)
        return None
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    return {"im_start": tok.convert_tokens_to_ids("<|im_start|>"),
            "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
            "think_end": tok.convert_tokens_to_ids("</think>"),
            "header": enc("assistant\n"),
            "blank": sorted(set(enc("\n") + enc("\n\n")))}


def answer_mask(tokens, markers: dict) -> np.ndarray:
    """True where a token belongs to an assistant answer."""
    tokens = np.asarray(tokens)
    out = np.zeros(len(tokens), dtype=bool)
    hdr, blank = list(markers["header"]), set(markers["blank"])
    ends = np.flatnonzero(tokens == markers["im_end"])
    for i in np.flatnonzero(tokens == markers["im_start"]):
        j = int(i) + 1 + len(hdr)
        if tokens[i + 1: j].tolist() != hdr:
            continue
        # skip an empty think block at the start of the answer
        head = tokens[j: j + 8].tolist()
        if markers.get("think_end") in head:
            j += head.index(markers["think_end"]) + 1
            while j < len(tokens) and int(tokens[j]) in blank:
                j += 1
        n = np.searchsorted(ends, j)
        out[j: int(ends[n]) if n < len(ends) else len(tokens)] = True
    return out


class TokenDataset:
    """Random fixed-length windows over a token file; masked labels become -100."""

    def __init__(self, split: str, seq_len: int, lookahead: int = 0,
                 device: str = "cuda", seed: int = 0, mask=None):
        self.path = bin_path(split)
        if not self.path.exists():
            raise FileNotFoundError("missing token file: " + str(self.path))
        self.data = np.memmap(self.path, dtype=DTYPE, mode="r")
        self.seq_len = seq_len
        self.lookahead = lookahead
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = str(device)
        self.span = seq_len + 1 + lookahead
        self.n = len(self.data)
        if self.n <= self.span:
            raise ValueError("token file %s too small (%d tokens, need > %d)"
                             % (self.path.name, self.n, self.span))
        self.rng = np.random.default_rng(seed)
        self.mask = mask

    def __len__(self) -> int:
        return self.n

    def batch(self, bs: int):
        """x (B,T), y (B,T), extra (B,lookahead,T) or None."""
        ix = self.rng.integers(0, self.n - self.span, size=bs)
        blk = np.stack([self.data[i: i + self.span] for i in ix]).astype(np.int64)
        t = torch.from_numpy(blk)
        x = t[:, : self.seq_len]
        y = t[:, 1: self.seq_len + 1]
        extra = None
        if self.lookahead > 0:
            extra = torch.stack(
                [t[:, 2 + j: self.seq_len + 2 + j] for j in range(self.lookahead)],
                dim=1)
        if self.mask is not None:
            keep = torch.from_numpy(
                np.stack([self.mask[i: i + self.span] for i in ix]))
            y = y.masked_fill(~keep[:, 1: self.seq_len + 1], -100)
            if extra is not None:
                extra = extra.masked_fill(~torch.stack(
                    [keep[:, 2 + j: self.seq_len + 2 + j]
                     for j in range(self.lookahead)], dim=1), -100)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to("cuda", non_blocking=True)
            y = y.pin_memory().to("cuda", non_blocking=True)
            if extra is not None:
                extra = extra.pin_memory().to("cuda", non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            if extra is not None:
                extra = extra.to(self.device)
        return x, y, extra

    def seq_iter(self, n_seqs: int, seq_len: int):
        """Deterministic windows for eval; yields (start offset, tokens)."""
        rng = np.random.default_rng(12345)
        span = seq_len + 1
        for _ in range(n_seqs):
            i = int(rng.integers(0, max(1, self.n - span)))
            blk = self.data[i: i + span].astype(np.int64)
            if len(blk) < span:
                continue
            yield i, torch.from_numpy(blk)


class MixedDataset:
    """Rows drawn from several token files at fixed shares."""

    def __init__(self, parts: list, seed: int = 0):
        self.parts = [p for p, _ in parts]
        w = np.array([float(s) for _, s in parts])
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, r) -> None:
        # Reseed the parts too, so a resumed run does not replay windows.
        self._rng = r
        for ds in self.parts:
            ds.rng = np.random.default_rng(int(r.integers(2 ** 63)))

    def batch(self, bs: int):
        counts = self._rng.multinomial(bs, self.p)
        got = [ds.batch(int(c)) for ds, c in zip(self.parts, counts) if c]
        x = torch.cat([g[0] for g in got])
        y = torch.cat([g[1] for g in got])
        extra = (torch.cat([g[2] for g in got])
                 if got[0][2] is not None else None)
        return x, y, extra
