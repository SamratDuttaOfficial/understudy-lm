"""Single source of truth for every knob. Persisted to config.json."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from typing import Any, List

from .paths import P
from .util import read_json, write_json, sha1


@dataclass
class Config:
    # ---- target model (the one we draft for) --------------------------------
    target_ollama: str = "qwen3.5:4b"
    target_hf: str = "Qwen/Qwen3.5-4B"

    # ---- prompt sourcing ---------------------------------------------------
    prompt_datasets: List[str] = field(default_factory=lambda: [
        "HuggingFaceTB/smoltalk",
        "allenai/tulu-3-sft-mixture",
        "HuggingFaceFW/fineweb-edu",
    ])
    n_prompts: int = 40000
    prompt_min_chars: int = 24
    prompt_max_chars: int = 1200

    # ---- on-policy generation (ollama) ------------------------------------
    gen_max_tokens: int = 512
    gen_temperature: float = 0.7
    gen_top_p: float = 0.95
    gen_concurrency: int = 6
    gen_shard_size: int = 250
    gen_timeout_s: int = 600
    gen_ctx: int = 2048
    gen_think: bool = False    # Qwen3.5 reasoning mode: on = empty responses

    # ---- tokenisation ------------------------------------------------------
    seq_len: int = 1024
    val_fraction: float = 0.02

    # ---- general chat corpus for pretraining -------------------------------
    # the target's ~20M tokens alone overfit, so pretraining mixes these in
    corpus_datasets: List[str] = field(default_factory=lambda: [
        "HuggingFaceTB/smoltalk",
        "allenai/tulu-3-sft-mixture",
    ])
    corpus_tokens: int = 300_000_000      # 0 = pretrain on the target's data only

    # ---- draft architecture ------------------------------------------------
    # shares the target's vocab; at 248K that is ~95M embedding, ~13M trunk
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_mult: float = 2.75
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0

    # ---- pretraining -------------------------------------------------------
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 400
    max_steps: int = 12000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ckpt_every: int = 200
    eval_every: int = 500
    eval_batches: int = 40
    log_every: int = 10
    keep_last_ckpts: int = 3
    compile_model: bool = False
    amp_dtype: str = "bf16"
    pretrain_corpus_share: float = 0.75   # share of pretraining rows from the corpus
    early_stop_evals: int = 4     # stop after this many evals without a new best (0 = off)

    # ---- post-training (multi-step lookahead on on-policy data) -----------
    post_steps: int = 3000
    post_lr: float = 1.0e-4
    post_warmup: int = 100
    lookahead: int = 3            # train to predict t+1 .. t+lookahead
    post_answers_only: bool = True  # learn only the target's answers, not the prompts

    # ---- evaluation --------------------------------------------------------
    spec_k: int = 4               # draft length per round (capped at 1 + lookahead heads)
    eval_seqs: int = 200
    full_eval: bool = False       # load the target in 4-bit for a wall-clock bench
    bench_prompts: int = 12
    bench_new_tokens: int = 128

    # ---- misc --------------------------------------------------------------
    seed: int = 1337
    device: str = "cuda"
    auto_install: bool = True
    num_workers: int = 2

    # ------------------------------------------------------------------ io --
    @classmethod
    def load(cls) -> "Config":
        raw = read_json(P.config, {}) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        write_json(P.config, asdict(self))

    def as_dict(self) -> dict:
        return asdict(self)

    def apply(self, overrides: List[str]) -> List[str]:
        """Apply key=value overrides; returns what was set."""
        applied = []
        names = {f.name for f in fields(self)}
        for ov in overrides or []:
            if "=" not in ov:
                raise SystemExit("--set expects key=value, got: " + repr(ov))
            k, v = ov.split("=", 1)
            k = k.strip()
            if k not in names:
                raise SystemExit("unknown config key " + repr(k) + " (see --show-config)")
            cur = getattr(self, k)
            try:
                if isinstance(cur, bool):
                    nv: Any = str(v).lower() in ("1", "true", "yes", "y", "on")
                elif isinstance(cur, int):
                    nv = int(v)
                elif isinstance(cur, float):
                    nv = float(v)
                elif isinstance(cur, list):
                    nv = [s for s in (x.strip() for x in v.split(",")) if s]
                else:
                    nv = v
            except ValueError:
                raise SystemExit("bad value for " + k + ": " + repr(v))
            setattr(self, k, nv)
            applied.append(k + "=" + str(nv))
        return applied

    # --------------------------------------------------------- stage hashes --
    def stage_hash(self, stage: str) -> str:
        keys = STAGE_KEYS.get(stage, [])
        d = self.as_dict()
        return sha1({k: d.get(k) for k in keys})


# Only these fields invalidate a stage, so changing lr never re-runs generate.
STAGE_KEYS = {
    "env":       [],
    "models":    ["target_ollama", "target_hf"],
    "prompts":   ["prompt_datasets", "n_prompts", "prompt_min_chars",
                  "prompt_max_chars", "seed"],
    "generate":  ["target_ollama", "n_prompts", "gen_max_tokens", "gen_temperature", "gen_think",
                  "gen_top_p", "gen_shard_size", "gen_ctx", "seed"],
    "tokenize":  ["target_hf", "seq_len", "val_fraction"],
    "corpus":    ["target_hf", "corpus_datasets", "corpus_tokens"],
    "pretrain":  ["d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_mult",
                  "rope_theta", "tie_embeddings", "seq_len",
                  "batch_size", "grad_accum", "lr", "warmup_steps", "max_steps",
                  "weight_decay", "min_lr_ratio", "seed",
                  "pretrain_corpus_share", "early_stop_evals"],
    "posttrain": ["post_steps", "post_lr", "post_warmup", "lookahead", "seed",
                  "post_answers_only", "early_stop_evals"],
    "evaluate":  ["spec_k", "eval_seqs", "full_eval", "bench_prompts",
                  "bench_new_tokens"],
    "plots":     [],
    "export":    [],
}
