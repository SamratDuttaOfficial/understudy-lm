"""All project paths. Everything lives under the project root - nothing escapes."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class P:
    root      = ROOT
    data      = ROOT / "data"
    prompts   = ROOT / "data" / "prompts"
    gen       = ROOT / "data" / "gen"
    tokens    = ROOT / "data" / "tokens"
    hf        = ROOT / "hf_cache"
    ckpt      = ROOT / "checkpoints"
    pretrain  = ROOT / "checkpoints" / "pretrain"
    posttrain = ROOT / "checkpoints" / "posttrain"
    state     = ROOT / "state"
    logs      = ROOT / "logs"
    metrics   = ROOT / "logs" / "metrics"
    downloads = ROOT / "downloads"
    out       = ROOT / "out"
    plots     = ROOT / "out" / "plots"
    reports   = ROOT / "out" / "reports"
    export    = ROOT / "out" / "export"

    config    = ROOT / "config.json"
    manifest  = ROOT / "state" / "manifest.json"
    registry  = ROOT / "state" / "registry.json"

    ALL = (data, prompts, gen, tokens, hf, ckpt, pretrain, posttrain, downloads,
           state, logs, metrics, out, plots, reports, export)


def ensure_dirs() -> None:
    for d in P.ALL:
        d.mkdir(parents=True, exist_ok=True)


def pin_caches() -> None:
    """Force every third-party cache inside the project folder."""
    P.hf.mkdir(parents=True, exist_ok=True)
    for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        os.environ[k] = str(P.hf)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
