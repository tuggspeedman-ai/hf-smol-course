# hf-smol-course

I built [tinychat](https://github.com/tuggspeedman-ai/tinychat) to learn pre-training from scratch. Tokenizer, architecture, training loop, the loss curve going down. That filled the "how do you build a language model" half of the story.

This repo is the other half: **post-training**. SFT, preference alignment (DPO), and vision-language fine-tuning. The [Hugging Face Smol Fine-Tuning Language Models course](https://huggingface.co/learn/smol-course/unit0/1) is the curriculum; this repo is my working notebook through it.

By the end of the course I want a published, preference-aligned small model on the HF Hub. The code that produced it lives here.

---

## Status

**Unit 1 (SFT) is complete.** Working through Unit 2 (DPO) next.

| Unit | Topic | Status |
|---|---|---|
| U0 | Welcome / setup | Done |
| U1 | Supervised fine-tuning with SmolLM3 | Done; 2 published adapters |
| U2 | Preference alignment (DPO) | Next |
| U3 | Vision-language models (SmolVLM2 fine-tuning) | Pending |
| U4 | Coming Soon (per the course's own placeholder) | Pending |

---

## Published artifacts

Two LoRA adapters on the Hugging Face Hub, both fine-tuned from `HuggingFaceTB/SmolLM3-3B-Base`:

- **[tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora](https://huggingface.co/tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora)** is the main one. SFT on 12k summarization examples from SmolTalk2's `smol_summarize` split. Trained on an A100 80GB via HF Jobs, ~97 min, ~$4 of compute. Loss 1.03 → 0.56, eval 0.44. Built from `notebooks/unit1/exercise3_sft_lora.py`.
- **[tuggspeedman-ai/SmolLM3-3B-trl-cli-demo](https://huggingface.co/tuggspeedman-ai/SmolLM3-3B-trl-cli-demo)** is the same recipe reproduced via TRL's stock `sft.py` CLI on a smaller dataset. The course's Exercise 4 "production workflow" rep. Config in `configs/u1_ex4_sft.yaml`.

---

## How the repo is laid out

```
notebooks/unit1/                    Hands-on exercises for each unit
  exercise1_chat_templates.py         Chat template internals (no GPU)
  exercise2_dataset_processing.py     SmolTalk2 schema + GSM8K normalization (no GPU)
  exercise3_sft_lora.py               The SFT training script (main artifact)
  exercise3_sft_lora_completed.ipynb  Captured session output from the local smoke run

configs/
  u1_ex4_sft.yaml                   TRL CLI hyperparameter config (Ex4 production rep)

.claude/
  commands/                         Claude Code slash commands I use for this project
  rules/core.md                     Engineering and ML rules the codebase follows

CLAUDE.md                           Project-level instructions for Claude Code
Makefile                            Common commands (setup, smoke, hf-login)
pyproject.toml                      uv-managed Python project
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) as the package manager and Python 3.12.

```bash
git clone https://github.com/tuggspeedman-ai/hf-smol-course.git
cd hf-smol-course

# Install base deps
make setup

# Or with Jupyter for cell-by-cell notebook work
make setup-notebooks

# Smoke-test the env (loads SmolLM2-135M)
make smoke

# Authenticate with HF (needs an HF_TOKEN in .env or the prompt)
make hf-login
```

To reproduce the U1 SFT run locally on Mac (Metal/MPS, ~16 hours at 24 s/step):

```bash
uv run python notebooks/unit1/exercise3_sft_lora.py    # SMOKE=True default
SMOKE=false uv run python notebooks/unit1/exercise3_sft_lora.py
```

To reproduce it on cloud via HF Jobs (~97 min, ~$4 on `a100-large`):

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env SMOKE=false \
  notebooks/unit1/exercise3_sft_lora.py
```

---

## What I've learned so far

A few load-bearing lessons from Unit 1. The full set lives in `.claude/memory/`, which I keep private. These are the ones worth surfacing publicly:

- **The chat template lives in the tokenizer, not the model.** SmolLM3-3B-Base ships a tokenizer with `chat_template = None`. Use the instruct tokenizer for templating; the vocab is shared with the base so the embedding indices align directly.
- **`load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=X)` resolves the entire 66GB config before isolating a split.** A bare `load_dataset` for a 2 MB split started downloading 7+ GB before I killed it. Fix: `hf_hub_download` of the exact parquet file.
- **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` in transformers 5.6, not a token list.** `len()` returns 2 (the dict's key count), not the token count. Bit my length-distribution analysis until I caught it. Use `len(enc["input_ids"])`.
- **On HF Jobs, push the irreplaceable artifact to the Hub *before* any optional work.** I lost a full 97-minute scale run because the after-generation loop pushed the Hub push past the job timeout. Cell 10 now saves and pushes immediately, then runs the after-gen as wrapped best-effort.
- **`torch>=2.4` in a PEP 723 inline-deps block silently resolves to a CUDA-13 wheel.** HF Jobs A100 hosts run CUDA-12 drivers. `torch.cuda.is_available()` returns `False` and training silently falls back to CPU. I burned $5 on a 2-hour A100 run that was actually running on CPU at 10 min/step. Fixed: pin `torch<2.11` in the PEP 723 block, hard-assert `device != "cpu"` in the script.

The codebase has comments explaining these in context, not just as anecdotes.

---

## What's still missing

- **U2 onward.** Everything past Unit 1.
- **A full fine-tune for comparison.** The current LoRA adapter is the local-feasible artifact. A full FT on cloud would be a fair comparison point for the model card.
- **A proper eval pass.** Right now I'm relying on train/eval loss + qualitative before/after generations. The course's leaderboard eval (via `hf jobs run`) is part of each unit's final-project submission, which I haven't tackled yet.
- **Sample efficiency exploration.** Higher LoRA rank, more epochs, larger subsample. I haven't ablated.

---

## Acknowledgements

- The [Hugging Face Smol Fine-Tuning Language Models course](https://huggingface.co/learn/smol-course/unit0/1) team for the curriculum.
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) by HuggingFaceTB.
- [TRL](https://github.com/huggingface/trl) for `SFTTrainer` and the CLI.

---

## License

Apache 2.0. See `LICENSE`.
