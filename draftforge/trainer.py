"""Resumable training loop. Checkpoints hold model, optimizer, step and RNG
state; metrics go to a JSONL file that the plots read."""
from __future__ import annotations
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import interrupt
from .data import TokenDataset
from .model import DraftLM, ModelArgs
from .util import LOG, append_jsonl, human_num, human_time, now_iso, write_json


def pick_device(want: str) -> torch.device:
    if want == "cuda" and not torch.cuda.is_available():
        LOG.warning("cuda requested but unavailable - falling back to cpu")
        return torch.device("cpu")
    return torch.device(want)


def amp_ctx(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dt = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    if dt is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dt = torch.float16
    return torch.autocast("cuda", dtype=dt)


def lr_at(step: int, base_lr: float, warmup: int, total: int,
          min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return base_lr * min_ratio
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def build_optimizer(model: DraftLM, lr: float, wd: float, device: torch.device):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = device.type == "cuda"
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                 fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def save_ckpt(path: Path, model: DraftLM, opt, step: int, cfg_dict: dict,
              run_id: str, best_val: float, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "step": step,
        "model_args": asdict(model.args),
        "config": cfg_dict,
        "run_id": run_id,
        "best_val": best_val,
        "saved_at": now_iso(),
        "np_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_ckpt(path: Path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def model_from_ckpt(blob: dict, device: torch.device,
                    n_extra_heads: Optional[int] = None) -> DraftLM:
    args = ModelArgs(**blob["model_args"])
    if n_extra_heads is not None:
        args.n_extra_heads = max(args.n_extra_heads, n_extra_heads)
    m = DraftLM(args).to(device)
    missing, unexpected = m.load_state_dict(blob["model"], strict=False)
    if missing:
        LOG.info("  %d new params not in checkpoint (expected when adding heads)",
                 len(missing))
    if unexpected:
        LOG.warning("  %d unexpected params in checkpoint", len(unexpected))
    return m


def prune_ckpts(d: Path, keep: int) -> None:
    ck = sorted(d.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ck[:-keep] if keep > 0 else []:
        try:
            p.unlink()
        except OSError:
            pass


@torch.no_grad()
def evaluate(model: DraftLM, ds: TokenDataset, batches: int, bs: int,
             device: torch.device, dtype_name: str) -> float:
    model.eval()
    # Fixed seed so val loss is comparable across steps and runs.
    ds.rng = np.random.default_rng(4242)
    tot = 0.0
    for _ in range(batches):
        x, y, extra = ds.batch(bs)
        with amp_ctx(device, dtype_name):
            _, loss, _ = model(x, y, extra)
        tot += float(loss)
    model.train()
    return tot / max(1, batches)


def _vram_report(device: torch.device) -> None:
    """Windows spills past VRAM into system RAM instead of failing, so warn."""
    peak = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    LOG.info("gpu memory: peak %.2f GB of %.2f GB", peak / 1e9, total / 1e9)
    if peak > total:
        LOG.warning("training needs more than the GPU's own memory and is "
                    "spilling into shared system RAM - it will be very slow. "
                    "Lower batch_size and raise grad_accum by the same factor.")


def train(*, cfg, **kw) -> dict:
    """Run or resume one training run; returns a summary dict."""
    try:
        return _train(cfg=cfg, **kw)
    except torch.cuda.OutOfMemoryError as e:
        bs, accum = cfg.batch_size, cfg.grad_accum
        raise RuntimeError(
            "out of GPU memory at batch_size=%d. Lower batch_size and raise "
            "grad_accum by the same factor - the effective batch stays the "
            "same: python run.py --set batch_size=%d --set grad_accum=%d"
            % (bs, max(1, bs // 2), accum * 2)) from e


def _train(*, cfg, run_dir: Path, run_id: str, max_steps: int, base_lr: float,
           warmup: int, lookahead: int, init_from: Optional[Path],
           metrics_file: Path, tag: str, corpus_share: float = 0.0,
           answer_markers: Optional[dict] = None) -> dict:
    """corpus_share: fraction of rows from the general corpus.
    answer_markers: train and validate on the 4B's answers only."""
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    from .data import load_meta
    meta = load_meta()
    vocab = int(meta.get("vocab_size", 248320))

    train_ds = TokenDataset("train", cfg.seq_len, lookahead, str(device), cfg.seed)
    val_ds = TokenDataset("val", cfg.seq_len, lookahead, str(device), cfg.seed + 1)
    if answer_markers:
        from .data import answer_mask
        for ds in (train_ds, val_ds):
            ds.mask = answer_mask(ds.data, answer_markers)
        if not train_ds.mask.any() or not val_ds.mask.any():
            LOG.warning("no chat answers found in the token files - learning "
                        "every token instead")
            train_ds.mask = val_ds.mask = None
        else:
            LOG.info("learning answers only (%.0f%% of training tokens)",
                     100 * train_ds.mask.mean())
    if corpus_share > 0:
        from .data import MixedDataset
        corpus = TokenDataset("corpus", cfg.seq_len, lookahead, str(device),
                              cfg.seed + 2)
        train_ds = MixedDataset([(train_ds, 1 - corpus_share),
                                 (corpus, corpus_share)], cfg.seed)
        LOG.info("training rows: %.0f%% general corpus (%s tokens), %.0f%% "
                 "the 4B's data", 100 * corpus_share, human_num(len(corpus)),
                 100 * (1 - corpus_share))

    start_step, best_val, since_best = 0, float("inf"), 0
    model, opt = None, None

    # resume, else branch from a parent, else start fresh
    if last.exists():
        blob = load_ckpt(last, map_location="cpu")
        model = model_from_ckpt(blob, device, lookahead)
        opt = build_optimizer(model, base_lr, cfg.weight_decay, device)
        if blob.get("opt"):
            try:
                opt.load_state_dict(blob["opt"])
            except Exception as e:
                LOG.warning("optimizer state incompatible (%s) - starting fresh", e)
        start_step = int(blob.get("step", 0))
        best_val = float(blob.get("best_val", float("inf")))
        since_best = int((blob.get("extra") or {}).get("since_best", 0))
        if (blob.get("extra") or {}).get("stopped_early"):
            max_steps = start_step
        try:
            np.random.set_state(blob["np_rng"])
            torch.set_rng_state(blob["torch_rng"].cpu()
                                if hasattr(blob["torch_rng"], "cpu")
                                else blob["torch_rng"])
        except Exception:
            pass
        LOG.info("resuming %s at step %d/%d (best val %.4f)",
                 run_id, start_step, max_steps, best_val)

    elif init_from is not None and Path(init_from).exists():
        blob = load_ckpt(init_from, map_location="cpu")
        model = model_from_ckpt(blob, device, lookahead)
        model.add_extra_heads(lookahead)
        model.to(device)
        opt = build_optimizer(model, base_lr, cfg.weight_decay, device)
        LOG.info("branched %s from %s (parent step %d)", run_id,
                 Path(init_from).name, blob.get("step", -1))

    else:
        args = ModelArgs.from_config(cfg, vocab, n_extra_heads=lookahead)
        model = DraftLM(args).to(device)
        opt = build_optimizer(model, base_lr, cfg.weight_decay, device)
        LOG.info("fresh init for %s", run_id)

    # A resumed run must not replay batches it already trained on.
    train_ds.rng = np.random.default_rng(cfg.seed + start_step)

    rep = model.param_report()
    LOG.info("draft model: %s params total (trunk %s, embedding %s, heads %s)",
             human_num(rep["total"]), human_num(rep["trunk"]),
             human_num(rep["embedding"]), human_num(rep["extra_heads"]))
    write_json(run_dir / "model_info.json",
               {"run_id": run_id, "tag": tag, "params": rep,
                "model_args": asdict(model.args), "vocab_size": vocab})

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            LOG.info("torch.compile enabled")
        except Exception as e:
            LOG.warning("torch.compile failed (%s) - continuing eager", e)

    if start_step >= max_steps:
        LOG.info("%s already at %d steps - nothing to do", run_id, start_step)
        return _summary(run_id, run_dir, start_step, best_val, rep, tag)

    model.train()
    bs, accum = cfg.batch_size, cfg.grad_accum
    t_start, t_win = time.time(), time.time()
    tok_win = 0
    end_step = max_steps

    for step in range(start_step, max_steps):
        # Stop only between steps, where the saved state is consistent.
        if interrupt.stopping():
            save_ckpt(last, _raw(model), opt, step, cfg.as_dict(), run_id,
                      best_val, {"since_best": since_best})
            LOG.info("%s stopped at step %d/%d - checkpoint saved, nothing "
                     "lost; re-run to continue", run_id, step, max_steps)
            raise interrupt.Interrupted("training stopped at step %d" % step)

        lr = lr_at(step, base_lr, warmup, max_steps, cfg.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, y, extra = train_ds.batch(bs)
            with amp_ctx(device, cfg.amp_dtype):
                _, loss, _ = model(x, y, extra)
                loss = loss / accum
            loss.backward()
            loss_acc += float(loss.detach()) * accum
            tok_win += x.numel()

        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        opt.step()
        if step == start_step and device.type == "cuda":
            _vram_report(device)

        if (step + 1) % cfg.log_every == 0:
            dt = time.time() - t_win
            tps = tok_win / max(1e-6, dt)
            done = step + 1 - start_step
            left = max_steps - (step + 1)
            eta = (time.time() - t_start) / max(1, done) * left
            LOG.info("step %d/%d  loss %.4f  lr %.2e  |g| %.2f  %s tok/s  eta %s",
                     step + 1, max_steps, loss_acc / accum, lr, gn,
                     human_num(tps), human_time(eta))
            append_jsonl(metrics_file, {
                "ts": now_iso(), "t": time.time(), "run_id": run_id, "tag": tag,
                "step": step + 1, "loss": loss_acc / accum, "lr": lr,
                "grad_norm": gn, "tok_per_s": tps,
                "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                               if device.type == "cuda" else 0.0)})
            t_win, tok_win = time.time(), 0

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == max_steps:
            vl = evaluate(model, val_ds, cfg.eval_batches, bs, device, cfg.amp_dtype)
            ppl = math.exp(min(20.0, vl))
            LOG.info("  val loss %.4f  ppl %.2f", vl, ppl)
            append_jsonl(metrics_file, {
                "ts": now_iso(), "t": time.time(), "run_id": run_id, "tag": tag,
                "step": step + 1, "val_loss": vl, "val_ppl": ppl})
            if vl < best_val:
                best_val, since_best = vl, 0
                save_ckpt(best_path, _raw(model), opt, step + 1,
                          cfg.as_dict(), run_id, best_val, {"val_loss": vl})
                LOG.info("  new best -> %s", best_path.name)
            else:
                since_best += 1
            if cfg.early_stop_evals and since_best >= cfg.early_stop_evals:
                save_ckpt(last, _raw(model), opt, step + 1, cfg.as_dict(),
                          run_id, best_val,
                          {"since_best": since_best, "stopped_early": True})
                LOG.info("%s: no new best val loss in %d evals - stopping "
                         "early at step %d/%d (best %.4f)", run_id, since_best,
                         step + 1, max_steps, best_val)
                end_step = step + 1
                break

        if (step + 1) % cfg.ckpt_every == 0 or (step + 1) == max_steps:
            save_ckpt(last, _raw(model), opt, step + 1, cfg.as_dict(),
                      run_id, best_val, {"since_best": since_best})
            save_ckpt(run_dir / ("step_%06d.pt" % (step + 1)), _raw(model), None,
                      step + 1, cfg.as_dict(), run_id, best_val)
            prune_ckpts(run_dir, cfg.keep_last_ckpts)

    total = time.time() - t_start
    LOG.info("%s finished %d steps in %s", run_id, end_step - start_step,
             human_time(total))
    return _summary(run_id, run_dir, end_step, best_val, rep, tag,
                    train_seconds=total)


def _raw(model):
    return getattr(model, "_orig_mod", model)


def _summary(run_id, run_dir, step, best_val, rep, tag, train_seconds=0.0) -> dict:
    return {"run_id": run_id, "tag": tag, "dir": str(run_dir), "steps": step,
            "best_val_loss": None if best_val == float("inf") else best_val,
            "best_val_ppl": None if best_val == float("inf")
            else math.exp(min(20.0, best_val)),
            "params": rep, "train_seconds": train_seconds,
            "last": str(run_dir / "last.pt"), "best": str(run_dir / "best.pt")}
