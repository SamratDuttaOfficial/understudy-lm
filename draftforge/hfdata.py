"""Prompts and conversations streamed from public HF datasets, with a
built-in prompt generator as an offline fallback."""
from __future__ import annotations
import hashlib
import itertools
import random
from typing import Iterable, List

from . import interrupt
from .util import LOG

_TEXT_KEYS = ("prompt", "instruction", "question", "query", "input", "text",
              "content")


def _from_row(row: dict, max_chars: int) -> str | None:
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if isinstance(msgs, list) and msgs:
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from") or ""
            if str(role).lower() in ("user", "human"):
                c = m.get("content") or m.get("value") or ""
                if isinstance(c, str) and c.strip():
                    return c.strip()[:max_chars]
        first = msgs[0]
        if isinstance(first, dict):
            c = first.get("content") or first.get("value") or ""
            if isinstance(c, str) and c.strip():
                return c.strip()[:max_chars]
    for k in _TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_chars]
    return None


def _stream_dataset(name: str, want: int, min_chars: int, max_chars: int) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        LOG.warning("`datasets` not installed - skipping %s", name)
        return []

    # some datasets need an explicit config
    attempts: List[tuple] = []
    if name == "HuggingFaceTB/smoltalk":
        attempts = [(name, "all")]
    elif name == "HuggingFaceFW/fineweb-edu":
        attempts = [(name, "sample-10BT")]
    attempts.append((name, None))

    for ds_name, cfg in attempts:
        try:
            kw = {"split": "train", "streaming": True}
            ds = load_dataset(ds_name, cfg, **kw) if cfg else load_dataset(ds_name, **kw)
            out: List[str] = []
            for row in ds:
                interrupt.check("prompt sourcing stopped")
                p = _from_row(row, max_chars)
                if p and len(p) >= min_chars:
                    out.append(p)
                if len(out) >= want:
                    break
            if out:
                LOG.info("  %s%s -> %d prompts", ds_name,
                         ("/" + cfg) if cfg else "", len(out))
                return out
        except Exception as e:
            LOG.debug("  %s%s failed: %s", ds_name, cfg or "", e)
            continue
    LOG.warning("  %s -> unavailable, skipped", name)
    return []


# whole conversations, for the corpus
_ROLES = {"user": "user", "human": "user", "assistant": "assistant",
          "gpt": "assistant", "system": "system"}
_CONFIGS = {"HuggingFaceTB/smoltalk": "all",
            "HuggingFaceFW/fineweb-edu": "sample-10BT"}


def conversation(row: dict):
    """A row's chat as [{"role", "content"}], or None if it has no assistant turn."""
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if not isinstance(msgs, list) or not msgs:
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = _ROLES.get(str(m.get("role") or m.get("from") or "").lower())
        c = m.get("content", m.get("value"))
        if role is None or not isinstance(c, str) or not c.strip():
            return None
        out.append({"role": role, "content": c})
    return out if any(m["role"] == "assistant" for m in out) else None


def stream_rows(name: str, skip: int = 0):
    """Stream a dataset's training rows, skipping the first `skip`."""
    from datasets import load_dataset
    kw = {"split": "train", "streaming": True}
    cfg = _CONFIGS.get(name)
    ds = load_dataset(name, cfg, **kw) if cfg else load_dataset(name, **kw)
    return ds.skip(skip) if skip else ds


# offline fallback prompts
_SUBJECTS = [
    "a REST API rate limiter", "the Python GIL", "gradient checkpointing",
    "B-tree indexes", "TLS certificate pinning", "the CAP theorem",
    "reservoir sampling", "copy-on-write memory", "consistent hashing",
    "vector clocks", "the borrow checker", "JIT compilation",
    "database sharding", "CRDTs", "eventual consistency", "WAL logging",
    "cache eviction policies", "zero-copy IO", "lock-free queues",
    "columnar storage", "bloom filters", "quorum reads",
]
_FORMS = [
    "Explain {s} to someone who has never seen it before.",
    "Write a short Python example that demonstrates {s}.",
    "What are the three most common mistakes people make with {s}?",
    "Compare {s} against the obvious alternative. When would you not use it?",
    "Summarise {s} in exactly five bullet points.",
    "Debug this: my system uses {s} and it is slow under load. Where do I look?",
    "Write documentation for a junior engineer about {s}.",
    "Give a step-by-step walkthrough of how {s} works internally.",
    "What trade-offs does {s} force you to make?",
    "Design a small experiment that would show {s} working correctly.",
]


def _builtin(want: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    combos = [f.format(s=s) for f, s in itertools.product(_FORMS, _SUBJECTS)]
    rng.shuffle(combos)
    out = []
    while len(out) < want:
        for c in combos:
            out.append(c)
            if len(out) >= want:
                break
    LOG.warning("using %d built-in fallback prompts (no HF datasets reachable)",
                len(out))
    return out[:want]


def prompt_key(p: str) -> str:
    """Hash of a prompt, ignoring case and whitespace."""
    return hashlib.md5(" ".join(p.lower().split()).encode("utf-8")).hexdigest()


def _dedup(prompts: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for p in prompts:
        h = prompt_key(p)
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return out


def collect_prompts(datasets: List[str], n: int, min_chars: int,
                    max_chars: int, seed: int) -> List[str]:
    """~n deduplicated prompts spread across the sources."""
    per = max(1, int(n / max(1, len(datasets))) + n // 10)
    pool: List[str] = []
    for name in datasets:
        interrupt.check("prompt sourcing stopped")
        LOG.info("sourcing prompts from %s ...", name)
        pool.extend(_stream_dataset(name, per, min_chars, max_chars))
    pool = _dedup(pool)
    if len(pool) < n:
        pool.extend(_builtin(n - len(pool), seed))
        pool = _dedup(pool)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]
