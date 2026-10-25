#!/usr/bin/env python
"""Convert the newest export in out/export to a Hugging Face Llama checkpoint
in out/hf, so the draft loads with plain transformers."""
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from draftforge.config import Config
from draftforge.model import DraftLM, ModelArgs
from draftforge.paths import P


def newest_export() -> Path:
    dirs = sorted(d for d in P.export.iterdir() if (d / "model.safetensors").exists())
    if not dirs:
        sys.exit("no export found - run `python run.py` first")
    return dirs[-1]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_export()
    out = P.out / "hf"
    out.mkdir(parents=True, exist_ok=True)

    # transformers' assisted generation needs the target's vocab_size exactly
    target = Config.load().target_hf
    tok = AutoTokenizer.from_pretrained(target, cache_dir=str(P.hf))
    tcfg = AutoConfig.from_pretrained(target, cache_dir=str(P.hf))
    tcfg = getattr(tcfg, "text_config", tcfg)
    target_vocab = tcfg.vocab_size
    eos = [tok.eos_token_id]
    for i in ([tcfg.eos_token_id] if isinstance(tcfg.eos_token_id, int) else tcfg.eos_token_id or []):
        if i not in eos:
            eos.append(i)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

    args = ModelArgs(**json.loads((src / "model_args.json").read_text()))
    draft = DraftLM(args)
    draft.load_state_dict(load_file(src / "model.safetensors"), strict=False)
    draft.eval()

    hd = args.d_model // args.n_heads
    kw = dict(vocab_size=target_vocab, hidden_size=args.d_model,
              intermediate_size=draft.blocks[0].ff.w1.out_features,
              num_hidden_layers=args.n_layers, num_attention_heads=args.n_heads,
              num_key_value_heads=args.n_kv_heads, head_dim=hd,
              max_position_embeddings=args.max_seq_len, rms_norm_eps=1e-6,
              hidden_act="silu", attention_bias=False, mlp_bias=False,
              tie_word_embeddings=True, bos_token_id=None, eos_token_id=eos,
              pad_token_id=pad)
    if "rope_parameters" in inspect.signature(LlamaConfig.__init__).parameters:
        kw["rope_parameters"] = {"rope_type": "default", "rope_theta": args.rope_theta}
    else:
        kw["rope_theta"] = args.rope_theta
    hf = LlamaForCausalLM(LlamaConfig(**kw)).eval()

    # Resize the padding rows to the target's vocab; extra rows copy one the
    # draft learned never to predict.
    emb = draft.tok.weight.detach()
    n = min(emb.shape[0], target_vocab)
    sd = {"model.embed_tokens.weight": torch.cat([emb[:n], emb[-1:].expand(target_vocab - n, -1)]),
          "model.norm.weight": draft.norm.weight}
    names = {"n1": "input_layernorm", "n2": "post_attention_layernorm",
             "attn.wq": "self_attn.q_proj", "attn.wk": "self_attn.k_proj",
             "attn.wv": "self_attn.v_proj", "attn.wo": "self_attn.o_proj",
             "ff.w1": "mlp.gate_proj", "ff.w3": "mlp.up_proj", "ff.w2": "mlp.down_proj"}
    for i, b in enumerate(draft.blocks):
        for ours, theirs in names.items():
            mod = b
            for part in ours.split("."):
                mod = getattr(mod, part)
            sd["model.layers.%d.%s.weight" % (i, theirs)] = mod.weight
    missing, unexpected = hf.load_state_dict({k: v.detach() for k, v in sd.items()}, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    hf.tie_weights()

    # The converted model must match the draft exactly.
    x = torch.randint(0, len(tok), (2, 256))
    with torch.no_grad():
        ours = draft.lm_head(draft.backbone(x))
        theirs = hf(x).logits
        h = hf.model(x).last_hidden_state[:, -1]
        heads = {"heads.%d.weight" % j: m.fc.weight.detach().contiguous()
                 for j, m in enumerate(draft.extra_heads)}
        hs = [h] + [h + F.silu(h @ heads["heads.%d.weight" % j].T) for j in range(len(heads))]
        guesses = hf.lm_head(torch.stack(hs, 1)).argmax(-1)
        want = draft.draft(x, k=len(heads) + 1)
    diff = (ours[..., :n] - theirs[..., :n]).abs().max().item()
    same = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    print("max logit diff %.2e, argmax agreement %.4f, heads match %s"
          % (diff, same, bool((guesses == want).all())))
    assert diff < 1e-3 and same == 1.0 and (guesses == want).all()

    hf.save_pretrained(out)
    save_file(heads, out / "lookahead_heads.safetensors")
    tok.save_pretrained(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
