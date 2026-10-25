#!/usr/bin/env python
"""draftforge - train a speculative-decoding draft model from a target LLM's text.

Resumable: run `python run.py` again and it continues where it stopped.
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback

from draftforge import interrupt
from draftforge.config import Config
from draftforge.paths import P, ensure_dirs, pin_caches
from draftforge.state import Manifest, Registry
from draftforge.util import (LOG, human_bytes, human_time, dir_size, now_ts,
                             on_battery, read_json, rmtree, setup_logging)
from draftforge import stages as S

BAR = "-" * 74


def stale_reason(name: str, cfg: Config, man: Manifest):
    """Why a stage must run (None if it is done and current)."""
    return man.stale_reason(name, cfg.stage_hash(name), S.BY_NAME[name].inputs)


# ----------------------------------------------------------------- display --
def show_status(cfg: Config, man: Manifest, reg: Registry) -> None:
    print(BAR)
    print("%-4s%-10s %-9s %-20s %s" % ("", "STAGE", "STATUS", "WHEN", "DETAIL"))
    print(BAR)
    for name in S.ORDER:
        st = S.BY_NAME[name]
        e = man.entry(name)
        status = e.get("status", "pending")
        if status == "done" and stale_reason(name, cfg, man):
            status = "stale"
        when = (e.get("finished") or e.get("started") or "")[:19]
        meta = e.get("meta", {}) or {}
        bits = []
        for k in ("n_prompts", "shards", "tokens", "train_tokens", "steps",
                  "n_plots", "vocab_size"):
            if k in meta:
                bits.append("%s=%s" % (k, meta[k]))
        if name == "generate" and status != "done":
            pr = man.progress("generate")
            if pr.get("shards_done"):
                bits.append("shards_done=%s" % pr["shards_done"])
        detail = " ".join(bits)[:34] or st.desc[:34]
        mark = {"done": "[x]", "running": "[~]", "failed": "[!]",
                "stopped": "[>]", "stale": "[s]"}.get(status, "[ ]")
        print("%-4s%-10s %-9s %-20s %s" % (mark, name, status, when, detail))
    print(BAR)
    sizes = [("data/gen", P.gen), ("data/tokens", P.tokens),
             ("checkpoints", P.ckpt), ("hf_cache", P.hf), ("out", P.out)]
    print("disk: " + "  ".join("%s=%s" % (n, human_bytes(dir_size(p)))
                               for n, p in sizes))
    print(BAR)


def show_versions(reg: Registry) -> None:
    for kind in ("pretrain", "posttrain", "eval"):
        items = reg.by_kind(kind)
        if not items:
            continue
        print("\n%s:" % kind.upper())
        for i in items:
            star = "*" if i.get("active") else " "
            m = i.get("metrics", {}) or {}
            bits = []
            for k in ("val_loss", "projected_speedup", "measured_speedup",
                      "expected_accepted", "steps"):
                if m.get(k) is not None:
                    v = m[k]
                    bits.append("%s=%s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
            print(" %s %-34s %s  %s" % (star, i["id"], i["created"][:19],
                                        " ".join(bits)))
    print("\n(* = active. Change with --promote <id>.)")


def show_summary(cfg: Config, man: Manifest, reg: Registry) -> None:
    s = read_json(P.reports / "summary.json", {}) or {}
    print("\n" + BAR)
    print("  DRAFTFORGE - RESULTS")
    print(BAR)
    if not s:
        print("  No summary yet - the export stage has not run.")
    else:
        d = s.get("data", {}) or {}
        pa = s.get("draft_params", {}) or {}
        ev = s.get("latest_eval", {}) or {}
        evals = sorted(P.reports.glob("eval_*.json"))
        if evals:   # newest eval, even if export hasn't re-run
            e = read_json(evals[-1], {}) or {}
            fast = e.get("fast") or {}
            ev = {k: fast.get(k) for k in ("p_accept", "expected_accepted",
                                           "tokens_per_round", "projected_speedup")}
            ev["measured_speedup"] = (e.get("full") or {}).get("measured_speedup")
        print("  target model     : %s" % s.get("target_model"))
        print("  draft model      : d_model=%s layers=%s heads=%s vocab=%s"
              % (pa.get("d_model"), pa.get("n_layers"), pa.get("n_heads"),
                 pa.get("vocab_size")))
        print("  training tokens  : train=%s  val=%s"
              % (d.get("train_tokens"), d.get("val_tokens")))
        print("  active pretrain  : %s" % s.get("active_pretrain"))
        print("  active posttrain : %s" % s.get("active_posttrain"))
        print(BAR)
        if ev.get("p_accept"):
            print("  acceptance       : " + ", ".join(
                "t+%d %.3f" % (i + 1, v) for i, v in enumerate(ev["p_accept"])))
        if ev.get("tokens_per_round") is not None:
            print("  tokens per round : %.2f  (%.2f accepted guesses + 1 from the target)"
                  % (ev["tokens_per_round"], ev["expected_accepted"]))
        elif ev.get("expected_accepted") is not None:
            print("  expected run     : %.2f accepted/round" % ev["expected_accepted"])
        if ev.get("projected_speedup") is not None:
            print("  projected speedup: %.2fx" % ev["projected_speedup"])
        if ev.get("measured_speedup") is not None:
            print("  MEASURED speedup : %.2fx" % ev["measured_speedup"])
        print(BAR)
        print("  export           : %s" % s.get("export_dir"))
    pngs = sorted(P.plots.glob("*.png"))
    if pngs:
        print("  plots (%d)        : %s" % (len(pngs), P.plots))
        print("      dashboard    : %s" % (P.plots / "dashboard.png"))
    print("  logs             : %s" % P.logs)
    print(BAR)
    print("  re-run graphs    : python plots.py")
    print("  all versions     : python run.py --list")
    print("  new posttrain    : python run.py --post-from <id> --tag myidea")
    print(BAR + "\n")


# ------------------------------------------------------------- orchestrate --
def run_stage(name: str, cfg: Config, man: Manifest, reg: Registry,
              force: bool, **kw) -> bool:
    st = S.BY_NAME[name]
    h = cfg.stage_hash(name)
    if not force:
        reason = stale_reason(name, cfg, man)
        if reason is None:
            LOG.info("[skip] %-10s already done", name)
            return True
        LOG.info("[run ] %-10s (%s)", name, reason)
    else:
        LOG.info("[run ] %-10s (forced)", name)
        # start a new run instead of resuming the finished one
        man.d.pop("progress::" + name, None)
        man.save()

    man.mark_running(name, h)
    t0 = time.time()
    try:
        meta = st.run(cfg, man, reg, **kw) or {}
    except KeyboardInterrupt:
        man.mark_stopped(name)
        LOG.info("stopped during %s after %s - progress is saved, just re-run",
                 name, human_time(time.time() - t0))
        raise
    except Exception as e:
        man.mark_failed(name, repr(e))
        LOG.error("stage %s failed: %s", name, e)
        LOG.debug("%s", traceback.format_exc())
        raise
    # a stage stopped via the flag returns normally but is not done
    if interrupt.stopping():
        man.mark_stopped(name)
        LOG.info("stopped during %s after %s - progress is saved, just re-run",
                 name, human_time(time.time() - t0))
        raise interrupt.Interrupted(name)
    man.mark_done(name, h, meta if isinstance(meta, dict) else {})
    LOG.info("[done] %-10s in %s", name, human_time(time.time() - t0))
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog="""examples:
  python run.py                          run/resume everything
  python run.py --status                 what is done, what is left
  python run.py --stage generate         run one stage only
  python run.py --from pretrain          run from this stage onward
  python run.py --set max_steps=20000    change a knob (persisted)
  python run.py --list                   all timestamped versions
  python run.py --post-from pretrain_20260908-101500 --tag lr5e5
  python run.py --promote posttrain_20260909-020000
  python run.py --eval-ckpt posttrain_20260909-020000
  python run.py --clean generate         delete one stage's outputs
  python run.py --reset                  wipe everything and start over
  python plots.py                        regenerate graphs only
""")
    g = p.add_argument_group("what to run")
    g.add_argument("--stage", action="append", metavar="NAME",
                   help="run only this stage (repeatable)")
    g.add_argument("--from", dest="from_", metavar="NAME",
                   help="start at this stage and continue")
    g.add_argument("--until", metavar="NAME", help="stop after this stage")
    g.add_argument("--force", action="store_true",
                   help="re-run even if already marked done")
    g.add_argument("--skip-install", action="store_true",
                   help="never pip-install anything")

    g = p.add_argument_group("inspect")
    g.add_argument("--status", action="store_true", help="stage table and exit")
    g.add_argument("--summary", action="store_true", help="final results and exit")
    g.add_argument("--list", action="store_true",
                   help="list every timestamped version")
    g.add_argument("--show-config", action="store_true",
                   help="print the full config and exit")

    g = p.add_argument_group("versions")
    g.add_argument("--promote", metavar="ID",
                   help="make this checkpoint the active one")
    g.add_argument("--post-from", metavar="ID",
                   help="start a NEW post-training run from this checkpoint")
    g.add_argument("--tag", metavar="NAME", default="",
                   help="label for the new post-training run")
    g.add_argument("--eval-ckpt", metavar="ID",
                   help="evaluate a specific checkpoint")

    g = p.add_argument_group("graphs")
    g.add_argument("--plots", nargs="*", metavar="NAME",
                   help="regenerate graphs (optionally only these) and exit")
    g.add_argument("--plot-stamp", action="store_true",
                   help="write timestamped plot filenames instead of overwriting")

    g = p.add_argument_group("config and cleanup")
    g.add_argument("--set", action="append", metavar="KEY=VAL",
                   help="override a config value, persisted (repeatable)")
    g.add_argument("--clean", action="append", metavar="STAGE",
                   help="delete a stage's outputs; 'all' cleans every stage")
    g.add_argument("--reset", action="store_true",
                   help="delete ALL generated data, checkpoints and state")
    g.add_argument("--keep-cache", action="store_true",
                   help="with --reset, keep the downloaded hf_cache")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    interrupt.install()
    ensure_dirs()
    pin_caches()
    setup_logging(P.logs / ("run_%s.log" % now_ts()), args.verbose)

    cfg = Config.load()
    set_only = False
    if args.set:
        applied = cfg.apply(args.set)
        cfg.save()
        LOG.info("config updated: %s", ", ".join(applied))
        # --set alone only changes the config
        set_only = not any([args.stage, args.from_, args.until, args.force,
                            args.status, args.summary, args.list,
                            args.show_config, args.promote, args.post_from,
                            args.eval_ckpt, args.clean, args.reset,
                            args.plots is not None])
    if args.skip_install:
        cfg.auto_install = False
    cfg.save()

    man, reg = Manifest(), Registry()
    reg.prune_missing()

    if set_only:
        show_status(cfg, man, reg)
        LOG.info("config saved. run `python run.py` to continue the pipeline.")
        return 0

    # ---- inspect-only paths -------------------------------------------------
    if args.show_config:
        import json
        print(json.dumps(cfg.as_dict(), indent=2))
        return 0
    if args.status:
        show_status(cfg, man, reg)
        return 0
    if args.list:
        show_versions(reg)
        return 0
    if args.summary:
        show_summary(cfg, man, reg)
        return 0
    if args.plots is not None:
        from draftforge import plots
        plots.make(args.plots or None, stamp=args.plot_stamp)
        return 0

    # ---- destructive ops ----------------------------------------------------
    if args.reset:
        LOG.warning("resetting project")
        for d in (P.data, P.ckpt, P.state, P.out, P.metrics):
            rmtree(d)
        if not args.keep_cache:
            rmtree(P.hf)
        for f in P.logs.glob("*.jsonl"):
            f.unlink(missing_ok=True)
        ensure_dirs()
        LOG.info("reset complete - next run starts from scratch")
        return 0

    if args.clean:
        targets = S.ORDER if "all" in args.clean else args.clean
        for name in targets:
            if name not in S.BY_NAME:
                LOG.error("unknown stage %r (choices: %s)", name,
                          ", ".join(S.ORDER))
                return 2
            LOG.info("cleaning %s", name)
            S.BY_NAME[name].clean(cfg)
            man.clear(name)
        LOG.info("cleaned: %s", ", ".join(targets))
        return 0

    # ---- version operations -------------------------------------------------
    if args.promote:
        it = reg.resolve(args.promote)
        if not it:
            LOG.error("no such version: %s (see --list)", args.promote)
            return 2
        reg.promote(it["id"])
        show_versions(reg)
        return 0

    if args.post_from:
        it = reg.resolve(args.post_from)
        if not it:
            LOG.error("no such checkpoint: %s (see --list)", args.post_from)
            return 2
        man.progress("posttrain").pop("run_id", None)   # force a NEW run dir
        man.save()
        try:
            S.run_posttrain(cfg, man, reg, init_ref=it["id"], tag=args.tag)
            man.mark_done("posttrain", cfg.stage_hash("posttrain"))
            S.run_evaluate(cfg, man, reg)
            man.mark_done("evaluate", cfg.stage_hash("evaluate"))
            S.run_plots(cfg, man, reg)
        except KeyboardInterrupt:
            man.mark_stopped("posttrain")
            LOG.info("stopped - re-run the same --post-from command to resume")
            return 130
        show_versions(reg)
        return 0

    if args.eval_ckpt:
        try:
            S.run_evaluate(cfg, man, reg, ckpt_ref=args.eval_ckpt)
            S.run_plots(cfg, man, reg)
        except KeyboardInterrupt:
            LOG.info("stopped - nothing was written; re-run to evaluate")
            return 130
        show_summary(cfg, man, reg)
        return 0

    # ---- normal pipeline ----------------------------------------------------
    if args.stage:
        todo = []
        for n in args.stage:
            if n not in S.BY_NAME:
                LOG.error("unknown stage %r (choices: %s)", n, ", ".join(S.ORDER))
                return 2
            todo.append(n)
    else:
        todo = list(S.ORDER)
        if args.from_:
            if args.from_ not in S.ORDER:
                LOG.error("unknown stage %r", args.from_)
                return 2
            todo = todo[S.ORDER.index(args.from_):]
        if args.until:
            if args.until not in S.ORDER:
                LOG.error("unknown stage %r", args.until)
                return 2
            todo = [n for n in todo if S.ORDER.index(n) <= S.ORDER.index(args.until)]

    pending = [n for n in todo
               if args.force or stale_reason(n, cfg, man) is not None]
    if not pending:
        LOG.info("everything is already done - showing results")
        show_status(cfg, man, reg)
        show_summary(cfg, man, reg)
        return 0

    LOG.info("plan: %s (later stages re-run if what they read changes)",
             " -> ".join(pending))
    if on_battery() and set(pending) & {"generate", "pretrain", "posttrain",
                                        "evaluate"}:
        LOG.warning("running on BATTERY - the GPU is throttled several times "
                    "over. Plug the laptop in; everything will still work, "
                    "just far slower.")
    t0 = time.time()
    try:
        for name in todo:
            interrupt.check()
            run_stage(name, cfg, man, reg, force=args.force)
    except KeyboardInterrupt:
        LOG.info("stopped after %s. everything is saved - `python run.py` "
                 "picks up from here", human_time(time.time() - t0))
        show_status(cfg, man, reg)
        return 130
    except Exception:
        LOG.error("pipeline halted after %s - fix the error and re-run; "
                  "completed stages will be skipped", human_time(time.time() - t0))
        return 1

    LOG.info("pipeline complete in %s", human_time(time.time() - t0))
    show_status(cfg, man, reg)
    show_summary(cfg, man, reg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
