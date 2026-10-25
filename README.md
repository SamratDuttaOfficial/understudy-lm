# understudy-lm

A speculative-decoding draft model for **Qwen3.5-4B**, trained only on the text
the 4B writes. Each pass it guesses the next four tokens; the 4B checks them in
one step and keeps the ones it agrees with, so the output is unchanged.

## Results

| | Run 1 | Run 2 |
|---|---|---|
| Pretraining data | 4B answers only | 75% public chat + 25% 4B answers |
| Best validation loss | 3.68 (overfit after step 4K) | **3.16** (still falling at 12K) |
| Tokens per 4B pass | 1.52 | **1.58** |
| Projected speedup | 1.38x | **1.43x** |

| Training loss | Acceptance per guess |
|---|---|
| ![Loss](assets/val_loss.png) | ![Acceptance](assets/acceptance.png) |
| **Tokens produced per round** | **Projected speedup** |
| ![Round tokens](assets/round_tokens.png) | ![Speedup](assets/speedup.png) |

The speedup is projected from replayed acceptance and timed draft rounds. In
llama.cpp, which drafts one token per pass and has to restore Qwen3.5's state
after every rejection, the 4B ran slower with the draft (27 to 36 tok/s against
69 tok/s), so realising it needs a runtime that uses the lookahead heads.

## Weights

On Hugging Face: [samratduttaofficial/understudy-lm](https://huggingface.co/samratduttaofficial/understudy-lm).
It is a standard `LlamaForCausalLM` with the Qwen3.5 tokenizer, so no clone of
this repo is needed:

```python
from transformers import AutoModelForCausalLM
draft = AutoModelForCausalLM.from_pretrained("samratduttaofficial/understudy-lm")
```

The model card shows how to use it as an assistant model for Qwen3.5-4B and how
to get four guesses per pass from the lookahead heads. `python export_hf.py`
rebuilds that checkpoint from a local training run.

## How it is trained

1. **Generate:** Qwen3.5-4B (through Ollama) answers 40K prompts, giving 13.4M tokens.
2. **Corpus:** 305M tokens of public chat (SmolTalk, Tulu 3), minus those prompts.
3. **Pretrain:** a 109M-parameter decoder on 75% corpus / 25% 4B answers, 12K steps.
4. **Post-train:** add three lookahead heads, train on the 4B's answers only, 3K steps.
5. **Evaluate:** replay greedy speculative decoding over held-out 4B answers.

## Run

Needs Python 3.10+ and an NVIDIA GPU.

```bash
python run.py
```

That one command installs dependencies, pulls the 4B, generates data, trains,
evaluates and writes graphs. It resumes where it stopped, so Ctrl+C and run it
again any time. Training peaks at about 4 GB of GPU memory with the defaults;
on a smaller GPU, trade batch size for accumulation before the first run:

```bash
python run.py --set batch_size=4 --set grad_accum=16
```

```bash
python run.py --status    # progress
python run.py --summary   # results
python plots.py           # redraw graphs into out/plots
python selftest.py        # 1-minute check of the whole pipeline
```
