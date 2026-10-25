"""Pipeline stages. Each one is idempotent and resumable."""
from __future__ import annotations
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from . import interrupt
from .paths import P, ensure_dirs, pin_caches
from .state import Manifest, Registry
from .util import (LOG, append_jsonl, dir_size, human_bytes, human_num,
                   human_time, now_ts, read_json, read_jsonl, rmtree,
                   write_json)


@dataclass
class Stage:
    name: str
    desc: str
    run: Callable
    clean: Callable
    deps: tuple = ()
    inputs: tuple = ()    # stages it reads from; if one re-runs, this is stale


PROMPTS_FILE = P.prompts / "prompts.jsonl"


def _has_nvidia_gpu() -> bool:
    """True if nvidia-smi sees a GPU, whatever torch thinks."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _shard_path(i: int) -> Path:
    return P.gen / ("shard_%05d.jsonl" % i)


def _shard_done(i: int) -> Path:
    return P.gen / ("shard_%05d.done" % i)


def _pool_map(fn, items, workers: int):
    """Map fn over items on daemon threads, yielding results as they land.
    Not ThreadPoolExecutor: its shutdown would make Ctrl+C wait on a request."""
    todo: queue.Queue = queue.Queue()
    done: queue.Queue = queue.Queue()
    for it in items:
        todo.put(it)
    halt = threading.Event()

    def loop():
        while not (halt.is_set() or interrupt.stopping()):
            try:
                it = todo.get_nowait()
            except queue.Empty:
                return
            try:
                done.put(fn(it))
            except Exception:
                done.put(None)

    threads = [threading.Thread(target=loop, daemon=True, name="gen%d" % i)
               for i in range(max(1, workers))]
    try:
        for t in threads:
            t.start()
        left = len(items)
        while left > 0:
            try:
                yield done.get(timeout=0.5)
                left -= 1
            except queue.Empty:
                if interrupt.stopping():
                    return
                if not any(t.is_alive() for t in threads) and done.empty():
                    return
    finally:
        halt.set()


# ============================================================== 1. env =======
def run_env(cfg, man: Manifest, reg: Registry) -> dict:
    ensure_dirs()
    pin_caches()
    LOG.info("project root: %s", P.root)

    missing = []
    for mod, pkg in (("torch", "torch"), ("numpy", "numpy"),
                     ("transformers", "transformers"), ("datasets", "datasets"),
                     ("matplotlib", "matplotlib"), ("safetensors", "safetensors")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing and cfg.auto_install:
        LOG.info("installing missing packages: %s", ", ".join(missing))
        if "torch" in missing:
            LOG.info("installing torch with CUDA 12.8 wheels (large download)")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch",
                            "--index-url", "https://download.pytorch.org/whl/cu128"],
                           check=False)
            missing = [m for m in missing if m != "torch"]
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing],
                           check=False)
        # Ctrl+C also reaches pip; don't import a half-installed tree.
        interrupt.check("install interrupted - re-run to continue")
    elif missing:
        raise RuntimeError("missing packages: %s (pip install -r requirements.txt)"
                           % ", ".join(missing))

    import torch

    # CPU-only torch on a GPU machine trains ~50x slower: swap in the CUDA build.
    if not torch.cuda.is_available() and _has_nvidia_gpu():
        LOG.warning("torch %s has no CUDA support but this machine has an "
                    "NVIDIA GPU", torch.__version__)
        if cfg.auto_install:
            LOG.warning("installing the CUDA build (2.7 GB, one time)")
            # Not --force-reinstall: it upgrades fsspec past what datasets allows.
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                            "torch"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "install", "torch",
                            "--index-url",
                            "https://download.pytorch.org/whl/cu128"], check=False)
            interrupt.check("torch install interrupted - re-run to continue")
            subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
            raise RuntimeError(
                "torch was reinstalled with CUDA support - re-run `python "
                "run.py` to continue (nothing was lost)")
        raise RuntimeError(
            "CPU-only torch on a CUDA machine. Fix with:\n"
            "  pip uninstall -y torch\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cu128\n"
            "or pass --set device=cpu to accept the slowdown.")

    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 2),
                    capability="%d.%d" % (p.major, p.minor),
                    bf16=torch.cuda.is_bf16_supported())
        LOG.info("GPU: %s  %.1f GB  sm_%d%d  bf16=%s", p.name,
                 p.total_memory / 1e9, p.major, p.minor,
                 torch.cuda.is_bf16_supported())
    else:
        LOG.warning("no CUDA device - everything will run on CPU and be very slow")
    write_json(P.state / "env.json", info)
    return info


def clean_env(cfg) -> None:
    (P.state / "env.json").unlink(missing_ok=True)


# ============================================================ 2. models ======
def run_models(cfg, man: Manifest, reg: Registry) -> dict:
    from . import ollama
    pin_caches()
    ollama.ensure_model(cfg.target_ollama, auto_install=cfg.auto_install)

    LOG.info("fetching tokenizer for %s", cfg.target_hf)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.target_hf, cache_dir=str(P.hf))
    meta = {"vocab_size": len(tok), "target_hf": cfg.target_hf,
            "target_ollama": cfg.target_ollama,
            "has_chat_template": bool(getattr(tok, "chat_template", None)),
            "eos_token_id": tok.eos_token_id}
    write_json(P.state / "tokenizer.json", meta)
    LOG.info("tokenizer ok: vocab=%d chat_template=%s", meta["vocab_size"],
             meta["has_chat_template"])
    return meta


def clean_models(cfg) -> None:
    (P.state / "tokenizer.json").unlink(missing_ok=True)
    LOG.info("(ollama model + hf cache left in place; delete %s to reclaim %s)",
             P.hf, human_bytes(dir_size(P.hf)))


# =========================================================== 3. prompts ======
def run_prompts(cfg, man: Manifest, reg: Registry) -> dict:
    pin_caches()
    from .hfdata import collect_prompts
    prompts = collect_prompts(cfg.prompt_datasets, cfg.n_prompts,
                              cfg.prompt_min_chars, cfg.prompt_max_chars, cfg.seed)
    P.prompts.mkdir(parents=True, exist_ok=True)
    tmp = PROMPTS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"i": i, "p": p}, ensure_ascii=False) + "\n")
    tmp.replace(PROMPTS_FILE)
    n_shards = (len(prompts) + cfg.gen_shard_size - 1) // cfg.gen_shard_size
    LOG.info("%d prompts -> %d shards of %d", len(prompts), n_shards,
             cfg.gen_shard_size)
    return {"n_prompts": len(prompts), "n_shards": n_shards,
            "file": str(PROMPTS_FILE)}


def clean_prompts(cfg) -> None:
    rmtree(P.prompts)


# ========================================================== 4. generate ======
def _load_prompts() -> List[dict]:
    rows = read_jsonl(PROMPTS_FILE)
    if not rows:
        raise RuntimeError("no prompts found - run the prompts stage first")
    return rows


def run_generate(cfg, man: Manifest, reg: Registry) -> dict:
    from . import ollama
    ollama.ensure_server(auto_install=cfg.auto_install)
    ollama.ensure_model(cfg.target_ollama, auto_install=cfg.auto_install)
    P.gen.mkdir(parents=True, exist_ok=True)

    # Threads beyond the server's slots only queue and time out.
    slots = ollama.detect_parallel_slots(default=cfg.gen_concurrency)
    workers = max(1, min(cfg.gen_concurrency, slots))
    if workers < cfg.gen_concurrency:
        LOG.warning("backend serves %d request at a time - using %d worker(s) "
                    "instead of gen_concurrency=%d (extra threads would only "
                    "queue and time out)", slots, workers, cfg.gen_concurrency)

    rows = _load_prompts()
    size = cfg.gen_shard_size
    n_shards = (len(rows) + size - 1) // size
    prog = man.progress("generate")
    total_tok = int(prog.get("tokens", 0))
    t_stage = time.time()

    for si in range(n_shards):
        sp, sd = _shard_path(si), _shard_done(si)
        if sd.exists():
            continue
        if interrupt.stopping():
            break
        chunk = rows[si * size: (si + 1) * size]
        part = sp.with_suffix(".part")

        # resume a partial shard from its .part file
        have = {}
        for src in (sp, part):
            if src.exists():
                for r in read_jsonl(src):
                    if "i" in r and r.get("r"):
                        have[r["i"]] = r
        todo = [c for c in chunk if c["i"] not in have]
        if todo:
            LOG.info("shard %d/%d: %d prompts (%d already done)",
                     si + 1, n_shards, len(todo), len(have))
        t0, ntok = time.time(), 0

        def work(item):
            try:
                out = ollama.generate(
                    cfg.target_ollama, item["p"],
                    max_tokens=cfg.gen_max_tokens, temperature=cfg.gen_temperature,
                    top_p=cfg.gen_top_p, ctx=cfg.gen_ctx,
                    timeout=cfg.gen_timeout_s, seed=cfg.seed + item["i"],
                    think=cfg.gen_think)
                return {"i": item["i"], "p": item["p"], "r": out["response"],
                        "n": out["eval_count"]}
            except Exception as e:
                LOG.debug("prompt %d failed: %s", item["i"], e)
                return None

        results = []
        if todo:
            for k, r in enumerate(_pool_map(work, todo, workers)):
                if r and r["r"].strip():
                    results.append(r)
                    ntok += r.get("n", 0) or 0
                    append_jsonl(part, r)
                if (k + 1) % 50 == 0:
                    LOG.info("  %d/%d  (%.0f tok/s)", k + 1, len(todo),
                             ntok / max(1e-6, time.time() - t0))

        allr = list(have.values()) + results
        allr.sort(key=lambda r: r["i"])
        tmp = sp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in allr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(sp)

        # Retire the shard only if most of it succeeded; the rest retry next run.
        ok_ratio = len(allr) / max(1, len(chunk))
        if ok_ratio >= 0.9:
            part.unlink(missing_ok=True)
            sd.write_text(now_ts(), encoding="utf-8")
        elif interrupt.stopping():
            LOG.info("shard %d paused with %d/%d saved - resumes right here",
                     si, len(allr), len(chunk))
        else:
            LOG.warning("shard %d kept open: only %d/%d prompts succeeded "
                        "(%.0f%%) - the rest retry on the next run",
                        si, len(allr), len(chunk), 100 * ok_ratio)

        dt = time.time() - t0
        total_tok += ntok
        man.set_progress("generate", "tokens", total_tok)
        man.set_progress("generate", "shards_done", si + 1)
        append_jsonl(P.logs / "generate.jsonl", {
            "t": time.time(), "shard": si, "n": len(allr), "tokens": ntok,
            "seconds": dt, "tok_per_s": ntok / max(1e-6, dt),
            "cum_tokens": total_tok})
        if todo and not interrupt.stopping():
            eta = (time.time() - t_stage) / max(1, si + 1) * (n_shards - si - 1)
            LOG.info("shard %d done: %d samples, %s tok, %.0f tok/s, eta %s",
                     si, len(allr), human_num(ntok), ntok / max(1e-6, dt),
                     human_time(eta))
        if interrupt.stopping():
            break

    n_out = sum(1 for _ in P.gen.glob("shard_*.jsonl"))
    return {"shards": n_out, "tokens": total_tok,
            "bytes": human_bytes(dir_size(P.gen))}


def clean_generate(cfg) -> None:
    rmtree(P.gen)
    (P.logs / "generate.jsonl").unlink(missing_ok=True)


# ========================================================== 5. tokenize ======
def run_tokenize(cfg, man: Manifest, reg: Registry) -> dict:
    pin_caches()
    import numpy as np
    from transformers import AutoTokenizer
    from . import data as D

    tok = AutoTokenizer.from_pretrained(cfg.target_hf, cache_dir=str(P.hf))
    prog = man.progress("tokenize")
    consumed = set(prog.get("consumed", []))

    # crash recovery
    D.truncate_to("train", int(prog.get("train_tokens", 0)))
    D.truncate_to("val", int(prog.get("val_tokens", 0)))
    n_tr = int(prog.get("train_tokens", 0))
    n_va = int(prog.get("val_tokens", 0))
    every = max(2, int(round(1.0 / max(1e-6, cfg.val_fraction))))
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    shards = sorted(P.gen.glob("shard_*.jsonl"))
    if not shards:
        raise RuntimeError("no generated shards - run the generate stage first")

    def encode(prompt: str, resp: str) -> List[int]:
        if getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": resp}],
                    tokenize=False)
                return tok(text, add_special_tokens=False).input_ids
            except Exception:
                pass
        return tok(prompt + "\n\n" + resp, add_special_tokens=False).input_ids

    for sp in shards:
        if sp.name in consumed:
            continue
        if interrupt.stopping():
            LOG.info("tokenize paused after %d/%d shards - resumes here",
                     len(consumed), len(shards))
            break
        rows = read_jsonl(sp)
        tr_buf, va_buf = [], []
        for r in rows:
            if not r.get("r"):
                continue
            ids = encode(r.get("p", ""), r["r"])
            if len(ids) < 8:
                continue
            ids = ids + [eos]
            (va_buf if r.get("i", 0) % every == 0 else tr_buf).extend(ids)
        if tr_buf:
            n_tr += D.append_tokens("train", np.array(tr_buf, dtype=D.DTYPE))
        if va_buf:
            n_va += D.append_tokens("val", np.array(va_buf, dtype=D.DTYPE))
        consumed.add(sp.name)
        man.set_progress("tokenize", "consumed", sorted(consumed))
        man.set_progress("tokenize", "train_tokens", n_tr)
        man.set_progress("tokenize", "val_tokens", n_va)
        if len(consumed) % 20 == 0:
            LOG.info("tokenized %d/%d shards  train=%s val=%s",
                     len(consumed), len(shards), human_num(n_tr), human_num(n_va))

    # before the size check, so a Ctrl+C isn't reported as too little data
    interrupt.check("tokenize stopped - re-run to continue")

    if n_va < cfg.seq_len * 4:
        raise RuntimeError("validation split too small (%d tokens) - lower "
                           "val_fraction or generate more data" % n_va)

    # the target's full vocab, padded to a multiple of 128
    vocab = 128 * ((len(tok) + 127) // 128)
    meta = {"vocab_size": vocab, "tokenizer": cfg.target_hf,
            "tokenizer_len": len(tok),
            "train_tokens": n_tr, "val_tokens": n_va, "seq_len": cfg.seq_len,
            "dtype": "uint32", "shards": len(shards), "built": now_ts()}
    D.save_meta(meta)
    LOG.info("tokens: train %s  val %s  vocab %d", human_num(n_tr),
             human_num(n_va), vocab)
    return meta


def clean_tokenize(cfg) -> None:
    rmtree(P.tokens)


# ============================================================ 6. corpus ======
def run_corpus(cfg, man: Manifest, reg: Registry) -> dict:
    """Public chat conversations for pretraining; the 4B's data alone overfits."""
    pin_caches()
    import numpy as np
    from transformers import AutoTokenizer
    from . import data as D
    from . import hfdata

    if cfg.corpus_tokens <= 0:
        LOG.info("corpus_tokens=0 - pretraining will use the 4B's data only")
        return {"tokens": 0}
    tok = AutoTokenizer.from_pretrained(cfg.target_hf, cache_dir=str(P.hf))
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    prog = man.progress("corpus")
    total = int(prog.get("tokens", 0))
    D.truncate_to("corpus", total)          # crash recovery
    used = ({hfdata.prompt_key(r["p"]) for r in read_jsonl(PROMPTS_FILE)}
            if PROMPTS_FILE.exists() else set())
    quota = cfg.corpus_tokens // max(1, len(cfg.corpus_datasets))
    part_rows, log_every = 2000, 20_000_000

    for name in cfg.corpus_datasets:
        key = "ds::" + name
        st = dict(prog.get(key) or {"rows": 0, "tokens": 0})
        failures = 0
        while st["tokens"] < quota and not st.get("exhausted"):
            LOG.info("corpus: %s from row %d (%s of %s tokens)", name,
                     st["rows"], human_num(st["tokens"]), human_num(quota))
            try:
                # opening a stream costs ~25s, so one serves every part
                rows = iter(hfdata.stream_rows(name, skip=st["rows"]))
                while st["tokens"] < quota:
                    texts, seen, ended = [], 0, True
                    for row in rows:
                        interrupt.check("corpus download stopped")
                        seen += 1
                        conv = hfdata.conversation(row)
                        first = hfdata._from_row(row, cfg.prompt_max_chars)
                        if conv and not (first and hfdata.prompt_key(first) in used):
                            try:
                                texts.append(tok.apply_chat_template(conv, tokenize=False))
                            except Exception:
                                pass        # a role order the template refuses
                        if seen >= part_rows:
                            ended = False
                            break
                    ids = []
                    for t in (tok(texts, add_special_tokens=False).input_ids
                              if texts else []):
                        ids.extend(t)
                        ids.append(eos)
                    before = total
                    if ids:
                        total += D.append_tokens("corpus", np.array(ids, dtype=D.DTYPE))
                    st["rows"] += seen
                    st["tokens"] += len(ids)
                    if ended:
                        st["exhausted"] = True
                    # commit tokens and rows in one write
                    prog[key], prog["tokens"] = dict(st), total
                    man.save()
                    failures = 0
                    if total // log_every != before // log_every:
                        LOG.info("corpus: %s tokens (%s from %s)", human_num(total),
                                 human_num(st["tokens"]), name)
                    if ended:
                        break
            except Exception as e:
                failures += 1
                if failures > 3:
                    LOG.warning("corpus: giving up on %s after repeated errors "
                                "(%s) - keeping its %s tokens", name, e,
                                human_num(st["tokens"]))
                    break
                LOG.warning("corpus: %s failed (%s) - retrying in 30s", name, e)
                time.sleep(30)

    if total == 0:
        LOG.warning("no corpus could be downloaded - pretraining will use the "
                    "4B's data only")
    LOG.info("corpus: %s tokens", human_num(total))
    return {"tokens": total}


def clean_corpus(cfg) -> None:
    from .data import bin_path
    bin_path("corpus").unlink(missing_ok=True)


# ========================================================== 7. pretrain ======
def _new_run_dir(kind: str, tag: str = "") -> tuple:
    ts = now_ts()
    rid = "%s_%s%s" % (kind, ts, ("_" + tag) if tag else "")
    base = P.pretrain if kind == "pretrain" else P.posttrain
    return rid, base / rid


def _run_ident(stage: str, cfg, man: Manifest, extra: str = "") -> str:
    """Config and input versions; if they change, a new run starts."""
    ups = ["%s@%.0f" % (u, man.finished_at(u)) for u in BY_NAME[stage].inputs]
    return "|".join([cfg.stage_hash(stage)] + ups + [extra])


def _resume_or_new(kind: str, man: Manifest, ident: str, tag: str = "") -> tuple:
    """Reuse this stage's in-flight run dir so a crash resumes in place."""
    prog = man.progress(kind)
    rid = prog.get("run_id")
    if rid and prog.get("ident") == ident:
        base = P.pretrain if kind == "pretrain" else P.posttrain
        d = base / rid
        if d.exists():
            return rid, d
    rid, d = _new_run_dir(kind, tag)
    prog["run_id"], prog["ident"] = rid, ident
    man.save()
    return rid, d


def _free_gpu() -> None:
    from . import ollama
    ollama.unload_all()


def run_pretrain(cfg, man: Manifest, reg: Registry) -> dict:
    from .data import token_count
    from .trainer import train
    _free_gpu()
    rid, rdir = _resume_or_new("pretrain", man, _run_ident("pretrain", cfg, man))
    share = cfg.pretrain_corpus_share
    if share > 0 and token_count("corpus") < 1_000_000:
        LOG.warning("no corpus on disk - pretraining on the 4B's data only")
        share = 0.0
    metrics = P.metrics / (rid + ".jsonl")
    summ = train(cfg=cfg, run_dir=rdir, run_id=rid, max_steps=cfg.max_steps,
                 base_lr=cfg.lr, warmup=cfg.warmup_steps, lookahead=0,
                 init_from=None, metrics_file=metrics, tag="pretrain",
                 corpus_share=share)
    reg.add(rid, "pretrain", rdir / "best.pt", parent=None,
            metrics={"val_loss": summ["best_val_loss"],
                     "val_ppl": summ["best_val_ppl"], "steps": summ["steps"]},
            config={"d_model": cfg.d_model, "n_layers": cfg.n_layers,
                    "max_steps": cfg.max_steps, "lr": cfg.lr,
                    "corpus_share": share},
            note="pretrain")
    reg.promote(rid)
    write_json(rdir / "summary.json", summ)
    return summ


def clean_pretrain(cfg) -> None:
    rmtree(P.pretrain)


# ========================================================= 8. posttrain ======
def run_posttrain(cfg, man: Manifest, reg: Registry,
                  init_ref: Optional[str] = None, tag: str = "") -> dict:
    from .data import chat_markers
    from .trainer import train
    parent = reg.resolve(init_ref) if init_ref else reg.active("pretrain")
    if not parent:
        raise RuntimeError("no pretrain checkpoint found - run pretrain first")
    init = Path(parent["path"])
    if not init.exists():
        raise RuntimeError("checkpoint missing on disk: " + str(init))

    _free_gpu()
    rid, rdir = _resume_or_new(
        "posttrain", man, _run_ident("posttrain", cfg, man, parent["id"]), tag)
    metrics = P.metrics / (rid + ".jsonl")
    LOG.info("post-training %s from %s (lookahead=%d)", rid, parent["id"],
             cfg.lookahead)
    summ = train(cfg=cfg, run_dir=rdir, run_id=rid, max_steps=cfg.post_steps,
                 base_lr=cfg.post_lr, warmup=cfg.post_warmup,
                 lookahead=cfg.lookahead, init_from=init,
                 metrics_file=metrics, tag="posttrain",
                 answer_markers=chat_markers(cfg) if cfg.post_answers_only else None)
    summ["parent"] = parent["id"]
    reg.add(rid, "posttrain", rdir / "best.pt", parent=parent["id"],
            metrics={"val_loss": summ["best_val_loss"],
                     "val_ppl": summ["best_val_ppl"], "steps": summ["steps"]},
            config={"post_steps": cfg.post_steps, "post_lr": cfg.post_lr,
                    "lookahead": cfg.lookahead,
                    "answers_only": cfg.post_answers_only},
            note=tag or "posttrain")
    reg.promote(rid)
    write_json(rdir / "summary.json", summ)
    return summ


def clean_posttrain(cfg) -> None:
    rmtree(P.posttrain)


# ========================================================== 9. evaluate ======
def run_evaluate(cfg, man: Manifest, reg: Registry,
                 ckpt_ref: Optional[str] = None) -> dict:
    import torch
    from .data import chat_markers
    from .trainer import load_ckpt, model_from_ckpt, pick_device
    from . import specdec

    item = (reg.resolve(ckpt_ref) if ckpt_ref
            else (reg.active("posttrain") or reg.active("pretrain")))
    if not item:
        raise RuntimeError("no checkpoint to evaluate")
    path = Path(item["path"])
    if not path.exists():
        raise RuntimeError("checkpoint missing: " + str(path))

    _free_gpu()
    device = pick_device(cfg.device)
    blob = load_ckpt(path, map_location="cpu")
    model = model_from_ckpt(blob, device)
    LOG.info("evaluating %s (step %s)", item["id"], blob.get("step"))

    out = {"checkpoint": item["id"], "path": str(path),
           "step": blob.get("step"), "when": now_ts(),
           "params": model.param_report(),
           "fast": specdec.fast_eval(model, device, cfg, chat_markers(cfg))}

    if cfg.full_eval:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.target_hf, cache_dir=str(P.hf))
        rows = read_jsonl(PROMPTS_FILE)[: cfg.bench_prompts]
        out["full"] = specdec.full_bench(model, cfg, device, tok,
                                         [r["p"] for r in rows])

    rid = "eval_%s_%s" % (now_ts(), item["id"])
    f = P.reports / (rid + ".json")
    write_json(f, out)
    reg.add(rid, "eval", f, parent=item["id"],
            metrics={"projected_speedup": out["fast"]["projected_speedup"],
                     "expected_accepted": out["fast"]["expected_accepted"],
                     "measured_speedup": (out.get("full") or {}).get("measured_speedup")})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def clean_evaluate(cfg) -> None:
    for f in P.reports.glob("eval_*.json"):
        f.unlink(missing_ok=True)


# ============================================================ 10. plots ======
def run_plots(cfg, man: Manifest, reg: Registry) -> dict:
    from . import plots
    files = plots.make()
    return {"n_plots": len(files), "dir": str(P.plots)}


def clean_plots(cfg) -> None:
    rmtree(P.plots)


# ============================================================ 11. export =====
def export_weights(blob: dict, path: Path) -> None:
    """Save weights as safetensors. A tied lm_head.weight is dropped, since
    safetensors refuses shared tensors; loading tok.weight restores it."""
    import torch
    from safetensors.torch import save_file

    sd = {k: v.contiguous() for k, v in blob["model"].items()
          if isinstance(v, torch.Tensor)}
    if blob["model_args"].get("tie_embeddings"):
        sd.pop("lm_head.weight", None)
    save_file(sd, str(path))


def run_export(cfg, man: Manifest, reg: Registry) -> dict:
    from .trainer import load_ckpt

    item = reg.active("posttrain") or reg.active("pretrain")
    if not item:
        raise RuntimeError("nothing to export")
    blob = load_ckpt(Path(item["path"]), map_location="cpu")

    ts = now_ts()
    d = P.export / ("draft_%s_%s" % (ts, item["id"]))
    d.mkdir(parents=True, exist_ok=True)
    export_weights(blob, d / "model.safetensors")
    write_json(d / "model_args.json", blob["model_args"])
    write_json(d / "train_config.json", blob.get("config", {}))
    write_json(d / "provenance.json",
               {"checkpoint": item["id"], "parent": item.get("parent"),
                "step": blob.get("step"), "exported": ts,
                "metrics": item.get("metrics", {})})

    evals =sorted(P.reports.glob("eval_*.json"))
    last_eval = read_json(evals[-1], {}) if evals else {}

    summary = {
        "generated": ts,
        "target_model": cfg.target_ollama,
        "draft_params": blob["model_args"],
        "stages": {k: v.get("status") for k, v in
                   (read_json(P.manifest, {}) or {}).get("stages", {}).items()},
        "data": read_json(P.tokens / "meta.json", {}),
        "active_pretrain": (reg.active("pretrain") or {}).get("id"),
        "active_posttrain": (reg.active("posttrain") or {}).get("id"),
        "latest_eval": {
            "checkpoint": last_eval.get("checkpoint"),
            "projected_speedup": (last_eval.get("fast") or {}).get("projected_speedup"),
            "expected_accepted": (last_eval.get("fast") or {}).get("expected_accepted"),
            "tokens_per_round": (last_eval.get("fast") or {}).get("tokens_per_round"),
            "p_accept": (last_eval.get("fast") or {}).get("p_accept"),
            "measured_speedup": (last_eval.get("full") or {}).get("measured_speedup"),
        },
        "export_dir": str(d),
        "versions": {k: [i["id"] for i in reg.by_kind(k)]
                     for k in ("pretrain", "posttrain", "eval")},
    }
    write_json(P.reports / "summary.json", summary)
    write_json(d / "summary.json", summary)
    LOG.info("exported to %s (%s)", d, human_bytes(dir_size(d)))
    return {"dir": str(d), "bytes": dir_size(d)}


def clean_export(cfg) -> None:
    rmtree(P.export)
    (P.reports / "summary.json").unlink(missing_ok=True)


# ============================================================== registry =====
STAGES: List[Stage] = [
    Stage("env", "check deps, GPU, dirs", run_env, clean_env),
    Stage("models", "pull ollama target + tokenizer", run_models, clean_models, ("env",)),
    Stage("prompts", "download + dedup prompts", run_prompts, clean_prompts, ("env",)),
    Stage("generate", "on-policy generation from the 4B", run_generate,
          clean_generate, ("models", "prompts")),
    Stage("tokenize", "encode to uint32 token bins", run_tokenize,
          clean_tokenize, ("generate",)),
    Stage("corpus", "download general chat text for pretraining", run_corpus,
          clean_corpus, ("models", "prompts")),
    Stage("pretrain", "train the draft model", run_pretrain, clean_pretrain,
          ("tokenize", "corpus"), inputs=("tokenize", "corpus")),
    Stage("posttrain", "add lookahead heads, fine-tune", run_posttrain,
          clean_posttrain, ("pretrain",), inputs=("pretrain",)),
    Stage("evaluate", "acceptance rate + speedup", run_evaluate, clean_evaluate,
          ("posttrain",), inputs=("posttrain",)),
    Stage("plots", "render all graphs", run_plots, clean_plots, (),
          inputs=("pretrain", "posttrain", "evaluate")),
    Stage("export", "safetensors + summary report", run_export, clean_export,
          ("evaluate",), inputs=("evaluate",)),
]

BY_NAME = {s.name: s for s in STAGES}
ORDER = [s.name for s in STAGES]
