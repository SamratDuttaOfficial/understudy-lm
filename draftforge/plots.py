"""Graphs, rendered from files on disk only. Add one with a `_p_<name>(ctx, ax)`
function registered in PLOTS."""
from __future__ import annotations
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import P  # noqa: E402
from .state import Registry  # noqa: E402
from .util import LOG, read_json, read_jsonl, now_ts  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False,
})
CYCLE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
         "#be185d", "#65a30d"]


class Ctx:
    """Everything the plot functions may read."""

    def __init__(self) -> None:
        self.reg = Registry()
        self.metrics: Dict[str, List[dict]] = {}
        for f in sorted(P.metrics.glob("*.jsonl")):
            rows = read_jsonl(f)
            if rows:
                self.metrics[f.stem] = rows
        self.evals: Dict[str, dict] = {}
        for f in sorted(P.reports.glob("eval_*.json")):
            d = read_json(f)
            if d:
                self.evals[f.stem] = d
        self.gen_stats = read_jsonl(P.logs / "generate.jsonl")
        self.summary = read_json(P.reports / "summary.json", {}) or {}

    def train_rows(self, name: str) -> List[dict]:
        return [r for r in self.metrics.get(name, []) if "loss" in r]

    def val_rows(self, name: str) -> List[dict]:
        return [r for r in self.metrics.get(name, []) if "val_loss" in r]


def _short(name: str) -> str:
    """'pretrain_20261023-211301' -> 'pretrain 10-23'. An eval is labelled
    with the checkpoint it scored."""
    if name.startswith("eval_"):
        name = name.split("_", 2)[-1]
    m = re.match(r"(.+?)_\d{4}(\d{2})(\d{2})-\d{6}$", name)
    return "%s %s-%s" % m.groups() if m else name


def _legend(ax) -> None:
    """Outside the axes, so it never covers data or labels."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)


def _ema(xs: List[float], a: float = 0.1) -> List[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else (a * x + (1 - a) * m)
        out.append(m)
    return out


def _nodata(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color="#888")
    ax.set_xticks([])
    ax.set_yticks([])


def _lines(ctx: Ctx, ax, rows_of, y, smooth=None, **kw) -> bool:
    any_ = False
    for i, name in enumerate(sorted(ctx.metrics)):
        rows = rows_of(name)
        if not rows:
            continue
        any_ = True
        c = CYCLE[i % len(CYCLE)]
        s, v = [r["step"] for r in rows], [r[y] for r in rows]
        if smooth:
            ax.plot(s, v, color=c, alpha=0.18, lw=0.8)
            v = _ema(v, smooth)
        ax.plot(s, v, color=c, label=_short(name), **kw)
    if any_:
        _legend(ax)
    return any_


# ------------------------------------------------------------------ plots ----
def _p_train_loss(ctx: Ctx, ax) -> None:
    if not _lines(ctx, ax, ctx.train_rows, "loss", smooth=0.1, lw=2):
        _nodata(ax, "no training metrics yet")
    ax.set(xlabel="step", ylabel="train loss", title="Training loss")


def _p_val_loss(ctx: Ctx, ax) -> None:
    if not _lines(ctx, ax, ctx.val_rows, "val_loss", marker="o", ms=4, lw=2):
        _nodata(ax, "no validation points yet")
    ax.set(xlabel="step", ylabel="val loss", title="Validation loss")


def _p_val_ppl(ctx: Ctx, ax) -> None:
    rows_of = lambda n: [r for r in ctx.val_rows(n) if r.get("val_ppl")]  # noqa: E731
    if not _lines(ctx, ax, rows_of, "val_ppl", marker="o", ms=4, lw=2):
        _nodata(ax, "no perplexity yet")
    ax.set(xlabel="step", ylabel="perplexity", title="Validation perplexity",
           yscale="log")


def _p_lr(ctx: Ctx, ax) -> None:
    rows_of = lambda n: [r for r in ctx.train_rows(n) if "lr" in r]  # noqa: E731
    if not _lines(ctx, ax, rows_of, "lr", lw=2):
        _nodata(ax, "no lr history yet")
    ax.set(xlabel="step", ylabel="learning rate", title="Learning-rate schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _p_throughput(ctx: Ctx, ax) -> None:
    rows_of = lambda n: [r for r in ctx.train_rows(n) if r.get("tok_per_s")]  # noqa: E731
    if not _lines(ctx, ax, rows_of, "tok_per_s", smooth=0.2, lw=2):
        _nodata(ax, "no throughput samples yet")
    ax.set(xlabel="step", ylabel="tokens/s", title="Training throughput")


def _p_gpu_mem(ctx: Ctx, ax) -> None:
    rows_of = lambda n: [r for r in ctx.train_rows(n) if r.get("gpu_mem_gb")]  # noqa: E731
    if not _lines(ctx, ax, rows_of, "gpu_mem_gb", lw=2):
        _nodata(ax, "no memory samples yet")
    ax.axhline(6.0, ls="--", lw=1.5, color="#dc2626")
    ax.text(0.02, 6.0, "6 GB card limit", color="#dc2626", ha="left",
            va="bottom", transform=ax.get_yaxis_transform())
    ax.set(xlabel="step", ylabel="GB", title="Peak GPU memory", ylim=(0, 6.8))


def _p_acceptance(ctx: Ctx, ax) -> None:
    ax.set_title("Acceptance by draft offset")
    if not ctx.evals:
        _nodata(ax, "no eval runs yet")
        return
    names = sorted(ctx.evals)[-4:]
    w = 0.8 / len(names)
    k = 0
    for i, n in enumerate(names):
        p = (ctx.evals[n].get("fast") or {}).get("p_accept") or []
        k = max(k, len(p))
        ax.bar([j + i * w for j in range(len(p))], p, width=w,
               color=CYCLE[i % len(CYCLE)], label=_short(n))
    ax.set(xlabel="draft offset", ylabel="P(accept | earlier accepted)",
           ylim=(0, 1))
    if k:
        ax.set_xticks([j + 0.4 - w / 2 for j in range(k)])
        ax.set_xticklabels(["t+%d" % (j + 1) for j in range(k)])
    _legend(ax)


def _bars(ax, names, vals, color, fmt) -> None:
    ax.bar(range(len(names)), vals, 0.6, color=color)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt % v, ha="center", va="bottom")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_short(n).replace(" ", "\n") for n in names])
    ax.set_ylim(0, max(vals + [0]) * 1.2 or 1)


def _p_speedup(ctx: Ctx, ax) -> None:
    ax.set_title("Projected speedup vs the 4B alone")
    if not ctx.evals:
        _nodata(ax, "no eval runs yet")
        return
    names = sorted(ctx.evals)
    fast = [ctx.evals[n].get("fast") or {} for n in names]
    _bars(ax, names, [f.get("projected_speedup", 0) for f in fast], CYCLE[0], "%.2fx")
    ax.axhline(1.0, ls="--", lw=1.5, color="#666")
    ax.set_ylabel("speedup")


def _p_expected_run(ctx: Ctx, ax) -> None:
    ax.set_title("Tokens per speculative round")
    if not ctx.evals:
        _nodata(ax, "no eval runs yet")
        return
    names = sorted(ctx.evals)
    tpr = [(ctx.evals[n].get("fast") or {}).get("tokens_per_round", 0) for n in names]
    _bars(ax, names, tpr, CYCLE[2], "%.2f")
    ax.set_ylabel("tokens")


def _p_generation(ctx: Ctx, ax) -> None:
    ax.set_title("Data generation progress")
    if not ctx.gen_stats:
        _nodata(ax, "no generation log yet")
        return
    # Hours spent generating, not calendar time: the machine was off in between.
    rows = sorted(ctx.gen_stats, key=lambda r: r.get("t", 0))
    hrs, cum, h, s = [], [], 0.0, 0
    for r in rows:
        h += r.get("seconds", 0) / 3600
        s += r.get("tokens", 0)
        hrs.append(h)
        cum.append(s / 1e6)
    ax.plot(hrs, cum, lw=2, color=CYCLE[0])
    ax.set(xlabel="hours of generation", ylabel="million tokens")
    if len(rows) > 1 and hrs[-1] > 0:
        ax.text(0.03, 0.95, "%.0f tok/s avg" % (cum[-1] * 1e6 / (hrs[-1] * 3600)),
                transform=ax.transAxes, va="top", color="#555")


def _p_gen_rate(ctx: Ctx, ax) -> None:
    ax.set_title("Generation rate per shard")
    if not ctx.gen_stats:
        _nodata(ax, "no generation log yet")
        return
    rows = sorted(ctx.gen_stats, key=lambda r: r.get("t", 0))
    rate = [r.get("tok_per_s", 0) for r in rows]
    ax.plot(range(len(rate)), rate, lw=1.2, color=CYCLE[3], alpha=0.4)
    ax.plot(range(len(rate)), _ema(rate, 0.2), lw=2.2, color=CYCLE[3])
    ax.set(xlabel="shard", ylabel="tokens/s")


def _p_params(ctx: Ctx, ax) -> None:
    info = None
    for it in ctx.reg.by_kind("pretrain") + ctx.reg.by_kind("posttrain"):
        d = read_json(Path(it["path"]).parent / "model_info.json")
        if d:
            info = d
    if not info:
        _nodata(ax, "no model built yet")
        ax.set_title("Parameter budget")
        return
    p = info["params"]
    parts = [("embedding", p.get("embedding", 0)), ("lm_head", p.get("lm_head", 0)),
             ("trunk", p.get("trunk", 0)), ("lookahead\nheads", p.get("extra_heads", 0))]
    parts = [(n, v / 1e6) for n, v in parts if v > 0][::-1]
    ax.barh([n for n, _ in parts], [v for _, v in parts],
            color=CYCLE[: len(parts)][::-1])
    for i, (_, v) in enumerate(parts):
        ax.text(v, i, " %.1fM" % v, va="center")
    ax.set_xlim(0, max(v for _, v in parts) * 1.3)
    ax.set(xlabel="million parameters",
           title="Parameters (%.1fM total)" % (p.get("total", 0) / 1e6))


def _p_stage_times(ctx: Ctx, ax) -> None:
    ax.set_title("Stage wall-clock")
    stages = (read_json(P.manifest, {}) or {}).get("stages", {})
    names, durs = [], []
    for n, e in stages.items():
        if e.get("status") != "done":
            continue
        d = e.get("elapsed_s")            # older entries lack it: last session only
        if n == "generate" and ctx.gen_stats:
            d = sum(r.get("seconds", 0) for r in ctx.gen_stats)
        if d is None:
            try:
                d = (datetime.fromisoformat(e["finished"])
                     - datetime.fromisoformat(e["started"])).total_seconds()
            except Exception:
                continue
        names.append(n)
        durs.append(max(d, 0.01) / 3600.0)
    if not names:
        _nodata(ax, "no completed stages yet")
        return
    ax.barh(names, durs, color=CYCLE[4])
    for i, d in enumerate(durs):
        ax.text(d, i, " %.1f h" % d if d >= 0.1 else " %.0f min" % (d * 60),
                va="center")
    ax.set_xlim(0, max(durs) * 1.3)
    ax.set_xlabel("hours")


PLOTS: Dict[str, Callable] = {
    "train_loss": _p_train_loss,
    "val_loss": _p_val_loss,
    "val_ppl": _p_val_ppl,
    "lr": _p_lr,
    "throughput": _p_throughput,
    "gpu_mem": _p_gpu_mem,
    "acceptance": _p_acceptance,
    "speedup": _p_speedup,
    "expected_run": _p_expected_run,
    "generation": _p_generation,
    "gen_rate": _p_gen_rate,
    "params": _p_params,
    "stage_times": _p_stage_times,
}


def make(only: List[str] | None = None, stamp: bool = False) -> List[Path]:
    """Render plots and dashboard.png to out/plots."""
    P.plots.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    names = [n for n in (only or list(PLOTS)) if n in PLOTS]
    if only:
        unknown = [n for n in only if n not in PLOTS]
        if unknown:
            LOG.warning("unknown plot(s): %s (available: %s)",
                        ", ".join(unknown), ", ".join(PLOTS))
    written: List[Path] = []
    suffix = ("_" + now_ts()) if stamp else ""

    for n in names:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        try:
            PLOTS[n](ctx, ax)
        except Exception as e:
            LOG.warning("plot %s failed: %s", n, e)
            _nodata(ax, "error: %s" % e)
        out = P.plots / (n + suffix + ".png")
        fig.savefig(out)
        plt.close(fig)
        written.append(out)

    cols = 3
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows),
                             layout="constrained")
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, n in zip(flat, names):
        try:
            PLOTS[n](ctx, ax)
        except Exception as e:
            _nodata(ax, "error: %s" % e)
    for ax in flat[len(names):]:
        ax.axis("off")
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.15)
    fig.suptitle("draftforge - %s" % now_ts(), fontsize=20, fontweight="bold")
    dash = P.plots / ("dashboard" + suffix + ".png")
    fig.savefig(dash)
    plt.close(fig)
    written.append(dash)

    LOG.info("wrote %d plots to %s", len(written), P.plots)
    return written
