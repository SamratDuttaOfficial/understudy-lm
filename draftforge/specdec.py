"""Speculative-decoding evaluation.

Fast path: replay greedy rounds over the target's own validation answers and
time one draft round. Full path (full_eval=true): wall-clock bench vs a 4-bit 4B.
"""
from __future__ import annotations
import copy
import time
from typing import Optional

import numpy as np
import torch

from . import interrupt
from .data import TokenDataset, answer_mask
from .paths import P
from .util import LOG, on_battery, read_jsonl


def draft_k(model, spec_k: int) -> int:
    """Proposals per round: 1 + n_heads with heads, else spec_k."""
    n = len(getattr(model, "extra_heads", []))
    return max(1, min(spec_k, n + 1)) if n else max(1, spec_k)


def _runs(mask: np.ndarray) -> list:
    """[start, end) of every stretch of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def replay_rounds(pr: np.ndarray, seq: np.ndarray, s: int, e: int):
    """Greedy speculative rounds over seq[s:e]; pr[j, p] is guess j after seq[p].
    Returns (tokens, rounds, reached, accepted)."""
    k = pr.shape[0]
    reached, accepted = np.zeros(k), np.zeros(k)
    toks = rounds = 0
    p = s - 1
    while p + 1 < e:
        a = 0
        while a < k and p + 1 + a < e:
            reached[a] += 1
            if pr[a, p] != seq[p + 1 + a]:
                break
            accepted[a] += 1
            a += 1
        step = min(a + 1, e - p - 1)      # accepted guesses + the target's token
        toks += step
        rounds += 1
        p += step
    return toks, rounds, reached, accepted


@torch.no_grad()
def acceptance_profile(model, ds: TokenDataset, n_seqs: int, seq_len: int,
                       device, k: int, dtype_name: str,
                       mask: Optional[np.ndarray] = None) -> dict:
    """Replay greedy rounds over the val split. p_accept[j] is P(guess j
    accepted | earlier guesses accepted). Without heads, accepted chained
    guesses equal the target's text, so teacher-forced predictions suffice."""
    from .trainer import amp_ctx
    model.eval()
    n_heads = len(getattr(model, "extra_heads", []))
    k = draft_k(model, k)
    win = min(seq_len, model.args.max_seq_len)
    reached, accepted = np.zeros(k), np.zeros(k)
    toks = rounds = 0

    for start, seq in ds.seq_iter(n_seqs, win):
        # Unwind rather than report a partial profile.
        interrupt.check("evaluation stopped")
        x = seq[:win].unsqueeze(0).to(device)
        with amp_ctx(device, dtype_name):
            h = model.backbone(x)
            rows = [model.lm_head(h).argmax(-1)[0]]
            if n_heads:
                rows += [model.lm_head(hd(h)).argmax(-1)[0]
                         for hd in model.extra_heads[: k - 1]]
        rows = [r.cpu().numpy() for r in rows]
        pr = (np.stack(rows) if n_heads
              else np.stack([np.roll(rows[0], -j) for j in range(k)]))
        s_np = seq.numpy()
        segs = ([(1, len(s_np))] if mask is None
                else _runs(mask[start: start + len(s_np)]))
        for s, e in segs:
            t, r, re_, ac = replay_rounds(pr, s_np, max(1, int(s)), int(e))
            toks += t
            rounds += r
            reached += re_
            accepted += ac

    return {"p_accept": (accepted / np.maximum(reached, 1)).tolist(),
            "tokens_per_round": toks / max(1, rounds),
            "expected_accepted": float(accepted.sum()) / max(1, rounds),
            "rounds": rounds, "n_seqs": n_seqs, "k": k, "n_heads": n_heads,
            "mode": "heads" if n_heads else "chain",
            "scored": "answers" if mask is not None else "all tokens"}


def _cuda_graph(fn):
    """Capture fn as one CUDA graph and return its replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):            # warm up off the capture stream
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


@torch.no_grad()
def measure_round_cost(model, device, k: int, iters: int = 50) -> tuple:
    """(seconds, method) for one draft round: bf16, one new token per pass.
    On CUDA it is replayed as a CUDA graph; eager PyTorch at this size is
    bound by kernel launches, which a deployed draft does not pay."""
    m = copy.deepcopy(model).to(torch.bfloat16).eval()
    n_heads = len(m.extra_heads)
    x = torch.randint(0, m.args.vocab_size, (1, 1), device=device)

    def one_round():
        if n_heads:
            h = m.backbone(x)[:, -1]
            hs = torch.cat([h] + [m.extra_heads[i](h) for i in range(k - 1)])
            return m.lm_head(hs).argmax(-1)
        y = x
        for _ in range(k):
            y = m.lm_head(m.backbone(y)[:, -1:]).argmax(-1)
        return y

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    fn, method = one_round, "eager"
    if device.type == "cuda":
        try:
            fn, method = _cuda_graph(one_round), "cuda graph"
        except Exception as e:
            LOG.warning("CUDA graph capture failed (%s: %s) - timing the draft "
                        "eagerly, which overstates its cost", type(e).__name__, e)

    for _ in range(5):
        fn()
    runs = []
    for _ in range(3):                    # median of three
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        runs.append((time.perf_counter() - t0) / iters)
    del m, fn
    return sorted(runs)[1], method


def target_step_seconds() -> tuple:
    """Target seconds per token: median over `generate` shards, else an estimate."""
    f = P.logs / "generate.jsonl"
    rates = sorted(r["tok_per_s"] for r in (read_jsonl(f) if f.exists() else [])
                   if r.get("tok_per_s", 0) > 0)
    if rates:
        return (1.0 / rates[len(rates) // 2],
                "measured during generate (median of %d shards)" % len(rates))
    return 2.4e9 / 264e9, "estimate: 2.4 GB of 4-bit weights at 264 GB/s"


def fast_eval(model, device, cfg, markers: Optional[dict] = None) -> dict:
    ds = TokenDataset("val", cfg.seq_len, 0, str(device), cfg.seed + 7)
    mask = None
    if markers:
        mask = answer_mask(np.asarray(ds.data), markers)
        if not mask.any():
            LOG.warning("no chat answers found in the val split - scoring "
                        "every token instead")
            mask = None
    prof = acceptance_profile(model, ds, cfg.eval_seqs, cfg.seq_len, device,
                              cfg.spec_k, cfg.amp_dtype, mask)

    unplugged = on_battery()
    if unplugged:
        LOG.warning("running on battery - the draft's timing will come out "
                    "several times too slow. Plug in and re-run "
                    "`python run.py --stage evaluate --force`")
    cost, method = measure_round_cost(model, device, prof["k"])
    target, source = target_step_seconds()
    # A round is one draft round plus one target pass; at batch 1 verifying
    # k tokens costs about the same as decoding one.
    ratio = cost / target
    out = {**prof, "draft_round_s": cost, "draft_timing": method,
           "target_step_s": target,
           "target_source": source, "draft_cost_ratio": ratio,
           "projected_speedup": prof["tokens_per_round"] / (1.0 + ratio),
           "on_battery": unplugged}
    LOG.info("acceptance, given the earlier guesses were accepted: %s",
             ", ".join("t+%d=%.3f" % (i + 1, v)
                       for i, v in enumerate(prof["p_accept"])))
    LOG.info("%.2f tokens per round over %d rounds (%d guesses via %s, "
             "scoring %s)", prof["tokens_per_round"], prof["rounds"],
             prof["k"], prof["mode"], prof["scored"])
    LOG.info("draft round %.2f ms (%s) vs target step %.2f ms (%s) -> "
             "projected speedup %.2fx", cost * 1e3, method, target * 1e3,
             source, out["projected_speedup"])
    return out


def _load_target(cfg, device):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(
        cfg.target_hf, quantization_config=q, device_map={"": 0},
        dtype=torch.bfloat16, low_cpu_mem_usage=True)
    m.eval()
    return m


@torch.no_grad()
def full_bench(model, cfg, device, tokenizer, prompts: list[str]) -> Optional[dict]:
    """Wall-clock benchmark: baseline vs speculative decoding."""
    try:
        target = _load_target(cfg, device)
    except Exception as e:
        LOG.warning("full eval skipped (%s)", e)
        return None

    from .trainer import amp_ctx
    model.eval()
    k, n_new = draft_k(model, cfg.spec_k), cfg.bench_new_tokens
    base_t, base_n, spec_t, spec_n, accepted, rounds = 0.0, 0, 0.0, 0, 0, 0

    for pr in prompts[: cfg.bench_prompts]:
        interrupt.check("benchmark stopped")
        ids = tokenizer(pr, return_tensors="pt").input_ids.to(device)

        # baseline greedy
        cur = ids.clone()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        past = None
        nxt = cur
        for _ in range(n_new):
            o = target(nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1:].argmax(-1)
            cur = torch.cat([cur, nxt], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        base_t += time.time() - t0
        base_n += n_new

        # speculative
        cur = ids.clone()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        produced = 0
        while produced < n_new:
            ctx = cur[:, -model.args.max_seq_len:]
            with amp_ctx(device, cfg.amp_dtype):
                prop = model.draft(ctx, k)                     # (1, k)
            cand = torch.cat([cur, prop], dim=1)
            logits = target(cand).logits[:, cur.size(1) - 1: -1]
            tgt = logits.argmax(-1)                            # (1, k)
            n_ok = 0
            for i in range(prop.size(1)):
                if int(prop[0, i]) == int(tgt[0, i]):
                    n_ok += 1
                else:
                    break
            take = torch.cat([prop[:, :n_ok], tgt[:, n_ok:n_ok + 1]], dim=1)
            cur = torch.cat([cur, take], dim=1)
            produced += take.size(1)
            accepted += n_ok
            rounds += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        spec_t += time.time() - t0
        spec_n += produced

    del target
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_tps = base_n / max(1e-6, base_t)
    spec_tps = spec_n / max(1e-6, spec_t)
    out = {"baseline_tok_s": base_tps, "spec_tok_s": spec_tps,
           "measured_speedup": spec_tps / max(1e-6, base_tps),
           "mean_accepted_per_round": accepted / max(1, rounds),
           "rounds": rounds, "k": k,
           "baseline_seconds": base_t, "spec_seconds": spec_t}
    LOG.info("measured: baseline %.1f tok/s -> spec %.1f tok/s (%.2fx), "
             "mean accepted %.2f/%d", base_tps, spec_tps,
             out["measured_speedup"], out["mean_accepted_per_round"], k)
    return out
