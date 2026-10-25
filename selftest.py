#!/usr/bin/env python
"""End-to-end check of the pipeline on synthetic data (~1 min, writes only
to .selftest/).

  python selftest.py            # run it
  python selftest.py --keep     # keep .selftest/ for inspection
"""
from __future__ import annotations
import argparse
import shutil
import sys

import numpy as np

from draftforge.paths import P, ensure_dirs, pin_caches
from draftforge.util import setup_logging, now_ts

SANDBOX = P.root / ".selftest"
VOCAB = 512
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def make_tokens(cfg) -> None:
    """Toy language: token t is usually followed by (t*7+3) % VOCAB."""
    from draftforge import data as D
    rng = np.random.default_rng(0)
    for split, n in (("train", 400_000), ("val", 60_000)):
        seq = np.zeros(n, dtype=np.uint32)
        cur = 1
        for i in range(n):
            seq[i] = cur
            cur = (cur * 7 + 3) % VOCAB if rng.random() < 0.9 else int(rng.integers(0, VOCAB))
        D.append_tokens(split, seq)
    D.save_meta({"vocab_size": VOCAB, "train_tokens": 400_000,
                 "val_tokens": 60_000, "seq_len": cfg.seq_len,
                 "dtype": "uint32", "built": now_ts()})


def check_chunked_loss(dev) -> None:
    """Chunked loss must match plain cross-entropy in value and gradients."""
    import torch
    import torch.nn.functional as F
    from draftforge.model import linear_cross_entropy

    g = torch.Generator().manual_seed(0)
    V, D = 300, 32
    W = torch.randn(V, D, generator=g).to(dev).requires_grad_()
    hs = [torch.randn(2, 37, D, generator=g).to(dev).requires_grad_() for _ in range(2)]
    ts = [torch.randint(0, V, (2, 37), generator=g).to(dev) for _ in range(2)]
    ts[0][0, :5] = -100

    def grads():
        out = [t.grad.clone() for t in (W, *hs)]
        for t in (W, *hs):
            t.grad = None
        return out

    ref = sum(s * F.cross_entropy((h @ W.t()).reshape(-1, V), t.reshape(-1),
                                  ignore_index=-100)
              for h, t, s in zip(hs, ts, (1.0, 0.4)))
    (ref * 0.5).backward()
    g_ref = grads()
    got = linear_cross_entropy(hs, W, ts, [1.0, 0.4], chunk=16)
    (got * 0.5).backward()
    g_got = grads()
    check("chunked loss equals cross-entropy",
          abs(float(got.detach()) - float(ref.detach())) < 1e-4,
          "%.6f vs %.6f" % (float(got.detach()), float(ref.detach())))
    check("chunked loss gradients match",
          all(torch.allclose(a, b, rtol=1e-3, atol=1e-5) for a, b in zip(g_got, g_ref)),
          "max diff %.2e" % max(float((a - b).abs().max()) for a, b in zip(g_got, g_ref)))


def check_real_size_fits(dev) -> None:
    """Real-size (248K vocab) training steps must fit in free VRAM. Checks the
    peak, since Windows spills to system RAM instead of raising OOM; two steps
    so gradients and Adam state are live."""
    import torch
    from draftforge.config import Config
    from draftforge.model import DraftLM, ModelArgs
    from draftforge.trainer import amp_ctx, build_optimizer

    if dev.type != "cuda":
        print("  SKIP  real-size step fits in VRAM  (no cuda)")
        return
    real = Config.load()
    T, V = real.seq_len, 248320
    for kind, la in (("pretrain", 0), ("posttrain", real.lookahead)):
        torch.cuda.empty_cache()
        # free memory plus what this process already holds
        budget = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        model = DraftLM(ModelArgs.from_config(real, V, la)).to(dev)
        opt = build_optimizer(model, 1e-4, real.weight_decay, dev)
        try:
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                for _ in range(2):
                    x = torch.randint(0, V, (real.batch_size, T + 1 + la), device=dev)
                    extra = (torch.stack([x[:, 2 + j: T + 2 + j] for j in range(la)], 1)
                             if la else None)
                    with amp_ctx(dev, real.amp_dtype):
                        _, loss, _ = model(x[:, :T], x[:, 1: T + 1], extra)
                    loss.backward()
                opt.step()
            torch.cuda.synchronize()
            peak, err = torch.cuda.max_memory_reserved(dev), ""
        except torch.cuda.OutOfMemoryError:
            peak, err = budget + 1, " (OOM)"
        del model, opt
        check("real-size %s step fits in VRAM" % kind, peak <= budget,
              "batch %d x %d, peak %.2f of %.2f GB free%s"
              % (real.batch_size, T, peak / 1e9, budget / 1e9, err))
    torch.cuda.empty_cache()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep .selftest/ afterwards")
    args = ap.parse_args(argv)

    ensure_dirs()
    pin_caches()
    setup_logging(P.logs / ("selftest_%s.log" % now_ts()))

    # redirect every writable path into the sandbox
    shutil.rmtree(SANDBOX, ignore_errors=True)
    P.tokens = SANDBOX / "tokens"
    P.metrics = SANDBOX / "metrics"
    P.pretrain = SANDBOX / "pretrain"
    P.posttrain = SANDBOX / "posttrain"
    P.reports = SANDBOX / "reports"
    P.plots = SANDBOX / "plots"
    for d in (P.tokens, P.metrics, P.pretrain, P.posttrain, P.reports, P.plots):
        d.mkdir(parents=True, exist_ok=True)

    from draftforge.state import Manifest, Registry
    P.state = SANDBOX / "state"
    P.registry = P.state / "registry.json"
    P.manifest = P.state / "manifest.json"
    P.state.mkdir(parents=True, exist_ok=True)
    for nm in ("a", "b", "c"):
        (P.state / nm).write_text("x", encoding="utf-8")

    print("\n-- resume + versioning --")
    r = Registry()
    r.add("pretrain_20260101-000000", "pretrain", P.state / "a")
    r.add("posttrain_20260102-000000", "posttrain", P.state / "b",
          parent="pretrain_20260101-000000")
    r.add("posttrain_20260103-000000", "posttrain", P.state / "c",
          parent="pretrain_20260101-000000")
    check("active defaults to newest",
          r.active("posttrain")["id"] == "posttrain_20260103-000000")
    r.promote("posttrain_20260102-000000")
    check("promote picks an older version",
          r.active("posttrain")["id"] == "posttrain_20260102-000000")
    check("promote is scoped per kind",
          r.active("pretrain")["id"] == "pretrain_20260101-000000")
    check("resolve by unique substring",
          (r.resolve("20260103") or {}).get("id") == "posttrain_20260103-000000")
    check("ambiguous ref refused", r.resolve("posttrain_2026") is None)
    check("history kept, nothing overwritten", len(r.by_kind("posttrain")) == 2)
    check("registry survives reload",
          Registry().active("posttrain")["id"] == "posttrain_20260102-000000")

    m = Manifest()
    m.mark_done("generate", "hash1", {"tokens": 123})
    check("completed stage is skipped", m.is_done("generate", "hash1"))
    check("config change marks stage stale",
          m.stale_reason("generate", "hash2") == "config changed")
    m.set_progress("generate", "shards_done", 7)
    check("in-stage progress persists",
          Manifest().progress("generate")["shards_done"] == 7)

    cfg_probe = __import__("draftforge.config", fromlist=["Config"]).Config()
    h_before = cfg_probe.stage_hash("generate")
    cfg_probe.lr = 0.001
    check("changing lr does NOT invalidate generate",
          cfg_probe.stage_hash("generate") == h_before)
    cfg_probe.n_prompts = 999
    check("changing n_prompts DOES invalidate generate",
          cfg_probe.stage_hash("generate") != h_before)

    # a stage is stale once one of its inputs completes again
    m.mark_done("corpus", "hc")
    m.mark_done("pretrain", "hp")
    check("done stage stays done", m.stale_reason("pretrain", "hp", ("corpus",)) is None)
    m.d["stages"]["corpus"]["finished_t"] += 100
    check("an input re-running makes a stage stale",
          m.stale_reason("pretrain", "hp", ("corpus",)) == "corpus re-ran")

    m.mark_running("posttrain", "hx")
    m.d["stages"]["posttrain"]["t_start"] -= 100
    m.mark_stopped("posttrain")
    m.mark_running("posttrain", "hx")
    m.d["stages"]["posttrain"]["t_start"] -= 50
    m.mark_done("posttrain", "hx")
    el = m.entry("posttrain")["elapsed_s"]
    check("stage time sums every session", 149 < el < 152, "%.1fs" % el)

    # same inputs resume a run; changed inputs start a new one
    from draftforge import stages as S
    P.pretrain = SANDBOX / "pretrain"
    rid1, d1_ = S._resume_or_new("pretrain", m, "ident-a")
    d1_.mkdir(parents=True, exist_ok=True)
    rid2, _ = S._resume_or_new("pretrain", m, "ident-a")
    _time_mod = __import__("time")
    _time_mod.sleep(1.1)                      # run ids are timestamped
    rid3, _ = S._resume_or_new("pretrain", m, "ident-b")
    check("same inputs resume the same run", rid2 == rid1)
    check("changed inputs start a new run", rid3 != rid1, "%s -> %s" % (rid1, rid3))

    from draftforge import hfdata
    conv = hfdata.conversation({"conversations": [
        {"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]})
    check("chat rows normalised", conv == [{"role": "user", "content": "hi"},
                                           {"role": "assistant", "content": "hello"}])
    check("chats with no answer skipped", hfdata.conversation(
        {"messages": [{"role": "user", "content": "hi"}]}) is None)

    print("\n-- ctrl+c --")
    import time as _time
    from draftforge import interrupt
    from draftforge.stages import _pool_map

    interrupt.reset()
    got = sorted(x for x in _pool_map(lambda i: i * 2, list(range(20)), 4)
                 if x is not None)
    check("worker pool returns every result", got == [i * 2 for i in range(20)],
          "%d results" % len(got))

    interrupt.reset()
    interrupt.STOP.set()
    t_stop = _time.time()
    drained = list(_pool_map(lambda i: _time.sleep(30), list(range(50)), 4))
    dt_stop = _time.time() - t_stop
    check("stop does not wait on in-flight work", dt_stop < 3.0,
          "returned in %.2fs" % dt_stop)
    check("stopped pool yields nothing half-finished", drained == [])
    interrupt.reset()

    m.mark_stopped("generate")
    check("stopped stage is not skipped next run",
          not Manifest().is_done("generate", "hash1"))
    check("stopped stage keeps its progress",
          Manifest().progress("generate").get("shards_done") == 7)

    print("\n-- model + training --")
    import torch
    from draftforge.config import Config
    from draftforge.trainer import train, load_ckpt, model_from_ckpt, pick_device
    from draftforge import specdec

    cfg = Config()
    cfg.seq_len, cfg.d_model, cfg.n_layers = 128, 128, 2
    cfg.n_heads, cfg.n_kv_heads = 4, 2
    cfg.batch_size, cfg.grad_accum = 4, 1
    cfg.warmup_steps, cfg.eval_every, cfg.ckpt_every, cfg.log_every = 5, 20, 10, 10
    cfg.eval_batches, cfg.eval_seqs, cfg.spec_k = 5, 20, 4
    cfg.lr, cfg.post_lr = 1e-3, 5e-4
    cfg.compile_model = False

    print("\n=== draftforge self-test ===")
    print("device: %s  cuda=%s" % (cfg.device, torch.cuda.is_available()))

    make_tokens(cfg)
    from draftforge.data import TokenDataset, token_count
    check("token bins written", token_count("train") == 400_000,
          "train=%d val=%d" % (token_count("train"), token_count("val")))
    ds = TokenDataset("train", cfg.seq_len, 2, cfg.device, 0)
    x, y, extra = ds.batch(4)
    check("batch shapes", tuple(x.shape) == (4, 128) and tuple(y.shape) == (4, 128)
          and extra is not None and tuple(extra.shape) == (4, 2, 128),
          str(tuple(x.shape)) + " " + str(tuple(extra.shape)))

    # labels outside an answer become -100, lookahead labels too
    keep = (np.arange(ds.n) // 50) % 2 == 0
    ds.mask = keep
    ds.rng = np.random.default_rng(3)
    x, y, extra = ds.batch(8)
    ds.rng = np.random.default_rng(3)
    ix = ds.rng.integers(0, ds.n - ds.span, size=8)
    want = np.stack([keep[i: i + ds.span] for i in ix])
    yc, ec = y.cpu().numpy(), extra.cpu().numpy()
    ok = ((yc == -100) == ~want[:, 1:129]).all() and all(
        ((ec[:, j] == -100) == ~want[:, 2 + j: 130 + j]).all() for j in range(2))
    check("answers-only labels skip prompt tokens", bool(ok),
          "%.0f%% of labels kept" % (100 * (yc != -100).mean()))
    ds.mask = None

    # toy corpus is all 7s, which the toy language never produces
    from draftforge import data as D
    D.append_tokens("corpus", np.full(50_000, 7, dtype=np.uint32))
    mix = D.MixedDataset([(TokenDataset("train", 128, 0, cfg.device, 0), 0.25),
                          (TokenDataset("corpus", 128, 0, cfg.device, 1), 0.75)], 0)
    rows = [mix.batch(8)[0].cpu().numpy() for _ in range(60)]
    share = np.mean([(r == 7).all(axis=1).mean() for r in rows])
    check("corpus mixed in at its share", 0.68 < share < 0.82, "%.2f" % share)

    check_chunked_loss(pick_device(cfg.device))
    check_real_size_fits(pick_device(cfg.device))

    d1 = P.pretrain / "st_pre"
    s1 = train(cfg=cfg, run_dir=d1, run_id="st_pre", max_steps=30,
               base_lr=cfg.lr, warmup=5, lookahead=0, init_from=None,
               metrics_file=P.metrics / "st_pre.jsonl", tag="pretrain")
    check("pretrain ran", (d1 / "last.pt").exists() and s1["steps"] == 30)
    loss_a = s1["best_val_loss"]
    check("model learned", loss_a is not None and loss_a < np.log(VOCAB),
          "val_loss=%.3f vs random %.3f" % (loss_a, np.log(VOCAB)))

    step_before = load_ckpt(d1 / "last.pt")["step"]
    s1b = train(cfg=cfg, run_dir=d1, run_id="st_pre", max_steps=50,
                base_lr=cfg.lr, warmup=5, lookahead=0, init_from=None,
                metrics_file=P.metrics / "st_pre.jsonl", tag="pretrain")
    check("resume continues, not restarts",
          step_before == 30 and s1b["steps"] == 50,
          "resumed from %d -> %d" % (step_before, s1b["steps"]))

    s1c = train(cfg=cfg, run_dir=d1, run_id="st_pre", max_steps=50,
                base_lr=cfg.lr, warmup=5, lookahead=0, init_from=None,
                metrics_file=P.metrics / "st_pre.jsonl", tag="pretrain")
    check("re-run when complete is a no-op", s1c["steps"] == 50)

    d3 = P.pretrain / "st_stop"
    kw3 = dict(cfg=cfg, run_dir=d3, run_id="st_stop", base_lr=cfg.lr,
               warmup=2, lookahead=0, init_from=None,
               metrics_file=P.metrics / "st_stop.jsonl", tag="pretrain")
    interrupt.reset()
    train(max_steps=10, **kw3)
    interrupt.STOP.set()                      # as if Ctrl+C landed right here
    stopped = False
    try:
        train(max_steps=40, **kw3)
    except KeyboardInterrupt:
        stopped = True
    interrupt.reset()
    check("ctrl+c stops training by unwinding", stopped)
    check("ctrl+c leaves a checkpoint at the exact step",
          load_ckpt(d3 / "last.pt")["step"] == 10,
          "step=%s" % load_ckpt(d3 / "last.pt")["step"])
    s3 = train(max_steps=40, **kw3)
    check("training resumes after ctrl+c", s3["steps"] == 40)

    # lr=0: val never improves, so patience 2 at eval_every 5 stops at step 15
    from dataclasses import replace
    cfg_es = replace(cfg, eval_every=5, early_stop_evals=2)
    d4 = P.pretrain / "st_early"
    kw4 = dict(cfg=cfg_es, run_dir=d4, run_id="st_early", max_steps=50,
               base_lr=0.0, warmup=1, lookahead=0, init_from=None,
               metrics_file=P.metrics / "st_early.jsonl", tag="pretrain",
               corpus_share=0.5)
    s4 = train(**kw4)
    check("early stopping ends a stalled run", s4["steps"] == 15,
          "stopped at %s" % s4["steps"])
    check("an early-stopped run stays stopped", train(**kw4)["steps"] == 15)

    d2 = P.posttrain / "st_post"
    s2 = train(cfg=cfg, run_dir=d2, run_id="st_post", max_steps=20,
               base_lr=cfg.post_lr, warmup=2, lookahead=3,
               init_from=d1 / "best.pt",
               metrics_file=P.metrics / "st_post.jsonl", tag="posttrain")
    check("posttrain branched from parent", (d2 / "last.pt").exists())
    blob = load_ckpt(d2 / "best.pt")
    check("lookahead heads attached", blob["model_args"]["n_extra_heads"] == 3,
          "n_extra_heads=%s" % blob["model_args"]["n_extra_heads"])

    dev = pick_device(cfg.device)
    model = model_from_ckpt(blob, dev)
    prop = model.draft(torch.randint(0, VOCAB, (1, 64), device=dev), k=4)
    check("draft proposes k tokens", tuple(prop.shape) == (1, 4), str(tuple(prop.shape)))

    fast = specdec.fast_eval(model, dev, cfg)
    p = fast["p_accept"]
    check("acceptance computed", len(p) == 4 and all(0 <= v <= 1 for v in p),
          "p=" + ", ".join("%.3f" % v for v in p))
    check("acceptance beats chance", p[0] > 5.0 / VOCAB,
          "p(t+1)=%.3f vs chance %.4f" % (p[0], 1.0 / VOCAB))
    check("speedup projected", fast["projected_speedup"] > 0,
          "%.2fx" % fast["projected_speedup"])
    check("draft timed as a deployed draft runs",
          fast["draft_timing"] == ("cuda graph" if dev.type == "cuda" else "eager"),
          "%s, %.2f ms/round" % (fast["draft_timing"], fast["draft_round_s"] * 1e3))
    check("rounds replayed", fast["rounds"] > 0 and fast["tokens_per_round"] >= 1.0,
          "%.2f tokens/round over %d rounds" % (fast["tokens_per_round"], fast["rounds"]))

    # 9 target tokens, 3 guesses: all right -> 3 rounds, all wrong -> 9
    seq = np.arange(10)
    right = np.stack([np.roll(seq, -(1 + j)) for j in range(3)])
    t_ok, r_ok, _, _ = specdec.replay_rounds(right, seq, 1, 10)
    t_no, r_no, _, _ = specdec.replay_rounds(np.full((3, 10), -1), seq, 1, 10)
    check("round replay counts tokens and rounds",
          (t_ok, r_ok, t_no, r_no) == (9, 3, 9, 9),
          "right %d/%d, wrong %d/%d" % (t_ok, r_ok, t_no, r_no))

    # ids: 1 <|im_start|>  2 <|im_end|>  3 </think>  4 assistant  5 newline
    # 6 blank  8 <think>  7/10/11 prompt  20-22 answer
    mk = {"im_start": 1, "im_end": 2, "think_end": 3, "header": [4, 5], "blank": [5, 6]}
    stream = np.array([1, 7, 5, 10, 11, 2, 1, 4, 5, 8, 6, 3, 6, 20, 21, 22, 2, 9])
    got = stream[specdec.answer_mask(stream, mk)].tolist()
    check("answer mask keeps only the answer", got == [20, 21, 22], str(got))

    from draftforge.util import write_json
    write_json(P.reports / "eval_selftest.json",
               {"checkpoint": "st_post", "fast": fast})

    # tied embeddings share one tensor, which safetensors refuses
    from safetensors.torch import load_file
    from draftforge.model import DraftLM, ModelArgs
    from draftforge.stages import export_weights
    ex = SANDBOX / "export" / "model.safetensors"
    ex.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_weights(blob, ex)
        err = ""
    except Exception as e:
        err = str(e).strip().splitlines()[0]
    check("export writes safetensors", ex.exists(), err)
    if ex.exists():
        back = DraftLM(ModelArgs(**blob["model_args"])).to(dev).eval()
        missing, unexpected = back.load_state_dict(load_file(str(ex)), strict=False)
        probe = torch.randint(0, VOCAB, (1, 32), device=dev)
        with torch.no_grad():
            same = torch.equal(back(probe)[0], model.eval()(probe)[0])
        check("exported weights reload identically",
              same and not unexpected and missing == ["lm_head.weight"],
              "missing=%s unexpected=%s" % (missing, unexpected))

    from draftforge import plots
    import importlib
    importlib.reload(plots)          # pick up the redirected paths
    files = plots.make()
    check("plots rendered", len(files) >= 10 and all(f.exists() for f in files),
          "%d files" % len(files))

    print("\n=== %s ===" % ("ALL PASSED" if not FAILED
                            else "FAILED: " + ", ".join(FAILED)))
    if not args.keep:
        shutil.rmtree(SANDBOX, ignore_errors=True)
    else:
        print("sandbox kept at %s" % SANDBOX)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
