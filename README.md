# hf-smol-course

I built [tinychat](https://github.com/tuggspeedman-ai/tinychat) to learn pre-training from scratch. Tokenizer, architecture, training loop, the loss curve going down. That filled the "how do you build a language model" half of the story.

This repo is the other half: **post-training**. SFT, preference alignment (DPO), and vision-language fine-tuning. The [Hugging Face Smol Fine-Tuning Language Models course](https://huggingface.co/learn/smol-course/unit0/1) is the curriculum; this repo is my working notebook through it.

By the end of the course I want a published, preference-aligned small model on the HF Hub. The code that produced it lives here.

---

## Status

**Units 1 (SFT), 2 (DPO), and 3 (VLM SFT) are complete.** U4 is the course's "Coming Soon" slot.

| Unit | Topic | Status |
|---|---|---|
| U0 | Welcome / setup | Done |
| U1 | Supervised fine-tuning with SmolLM3 | Done; 2 published adapters |
| U2 | Preference alignment (DPO) | Done; 1 published adapter |
| U3 | Vision-language models (SmolVLM2 fine-tuning) | Done; 1 published adapter |
| U4 | Coming Soon (per the course's own placeholder) | Pending |

---

## Published artifacts

Four LoRA adapters on the Hugging Face Hub. The first three are fine-tuned from `HuggingFaceTB/SmolLM3-3B-Base`; the fourth from `HuggingFaceTB/SmolVLM2-2.2B-Instruct`:

- **[tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora](https://huggingface.co/tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora)** is the SFT base. SFT on 12k summarization examples from SmolTalk2's `smol_summarize` split. Trained on an A100 80GB via HF Jobs, ~97 min, ~$4 of compute. Loss 1.03 → 0.56, eval 0.44. Built from `notebooks/unit1/exercise3_sft_lora.py`.
- **[tuggspeedman-ai/SmolLM3-3B-summarize-dpo-lora](https://huggingface.co/tuggspeedman-ai/SmolLM3-3B-summarize-dpo-lora)** is the preference-aligned one, and the model the course is ultimately building toward. DPO on top of the SFT adapter above. It keeps training the *same* LoRA rather than starting a fresh one, with the pre-DPO adapter frozen as the reference policy. 12k preference pairs from SmolTalk2's Tulu 3 mix. A100 80GB via HF Jobs, ~2.4h, ~$6. Loss 0.70 → 0.59, eval reward accuracy 0.68, reward margin +0.47. Outputs are shorter and cleaner than the SFT model's, and it fixed a repetition loop the SFT model fell into on one prompt. Built from `notebooks/unit2/exercise2_dpo_lora.py`.
- **[tuggspeedman-ai/SmolLM3-3B-trl-cli-demo](https://huggingface.co/tuggspeedman-ai/SmolLM3-3B-trl-cli-demo)** is the same SFT recipe, but run through TRL's standard command-line tool instead of a custom Python script. The course used this as Exercise 4 to simulate a "production workflow", the CLI-driven setup a team would actually use. Trained on a smaller dataset as a quick proof of concept. Config in `configs/u1_ex4_sft.yaml`.
- **[tuggspeedman-ai/SmolVLM2-2.2B-chartqa-lora](https://huggingface.co/tuggspeedman-ai/SmolVLM2-2.2B-chartqa-lora)** is the vision-language sidetrack, the modality breadth piece. SFT on 2,830 chart Q&A examples from `HuggingFaceM4/ChartQA`. LoRA on the language model only, with the SigLIP vision encoder frozen (scoped via regex to `model.text_model.*`, since the textbook `q_proj`/`v_proj` list would otherwise also adapt the vision tower). A10G 24GB via HF Jobs, ~79 min, ~$2 of compute. Loss 0.745 → 0.219, eval token accuracy 0.82. The visible adaptation is the answer-format shift, from verbose paragraphs to single-word answers; factual chart reading is not much better than the base model already did. Built from `notebooks/unit3/exercise_vlm_sft.py`.

---

## How the repo is laid out

```
notebooks/unit1/                    Hands-on exercises for each unit
  exercise1_chat_templates.py         Chat template internals (no GPU)
  exercise2_dataset_processing.py     SmolTalk2 schema + GSM8K normalization (no GPU)
  exercise3_sft_lora.py               The SFT training script
  exercise3_sft_lora_completed.ipynb  Captured session output from the local smoke run
notebooks/unit2/
  exercise2_dpo_lora.py               The DPO training script (preference alignment)
notebooks/unit3/
  exercise_vlm_sft.py                 The VLM SFT training script (SmolVLM2 on ChartQA)

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

The U2 DPO run works the same way (~2.4h, ~$6 on `a100-large`). It loads the U1 SFT adapter, so that one needs to exist first:

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 4h \
  --secrets HF_TOKEN \
  --env SMOKE=false \
  notebooks/unit2/exercise2_dpo_lora.py
```

The U3 VLM SFT run uses `a10g-large` instead (cheaper, fits SmolVLM2-2.2B plus image tokens fine at batch 1; ~79 min, ~$2):

```bash
hf jobs uv run \
  --flavor a10g-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  --env SMOKE=false \
  notebooks/unit3/exercise_vlm_sft.py
```

---

## What I've learned so far

A few load-bearing lessons from Units 1 and 2. The full set lives in `.claude/memory/`, which I keep private. These are the ones worth surfacing publicly:

- **The chat template lives in the tokenizer, not the model.** SmolLM3-3B-Base ships a tokenizer with `chat_template = None`. Use the instruct tokenizer for templating; the vocab is shared with the base so the embedding indices align directly.
- **`load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=X)` resolves the entire 66GB config before isolating a split.** A bare `load_dataset` for a 2 MB split started downloading 7+ GB before I killed it. Fix: `hf_hub_download` of the exact parquet file.
- **On HF Jobs, push the irreplaceable artifact to the Hub *before* any optional work.** I lost a full 97-minute scale run because the after-generation loop pushed the Hub push past the job timeout. Cell 10 now saves and pushes immediately, then runs the after-gen as wrapped best-effort.
- **A loose `torch>=2.4` pin installs a CUDA-13 build, but HF Jobs A100s run CUDA-12.** On a mismatch, `torch.cuda.is_available()` is `False` and training quietly runs on CPU. I burned $5 on a 2-hour A100 run that was really on CPU at 10 min/step before I caught it. Fix: pin `torch<2.11`, and assert `device != "cpu"` before training starts.
- **A too-low DPO learning rate produces a run that looks healthy but never moves the model.** At the course's suggested 5e-7, my 50-step smoke trained without errors and the loss sat right at the expected starting point, but the before/after generations came out byte-identical. The adapter hadn't budged. Bumping to 1e-6 made the policy actually move (reward margin -0.01 → +0.41). The cheap before/after diff in the smoke is what caught it; the loss curve alone wouldn't have.
- **HF Jobs prepaid credit can run out mid-run, and the job dies before it saves anything.** My first DPO scale run got auto-canceled at 85% (epoch 0.86 of 1.0) the moment the balance hit zero. ~$6 spent, no model, because the cancel landed before the Hub push. The reason isn't in `hf jobs inspect`; it's in the `/api/jobs/<owner>/<id>` response as `cancelReason: NO_CREDITS`. Now I top up to roughly 2x the estimated cost before submitting.

The codebase has comments explaining these in context, not just as anecdotes.

---

## What's still missing

- **U4** whenever it ships (currently "Coming Soon" per the course's own placeholder).
- **A full fine-tune for comparison.** The current adapters are the local-feasible artifacts. A full FT on cloud would be a fair comparison point for the model cards.
- **A proper eval pass.** Right now I'm relying on train/eval loss, reward margins, and qualitative before/after generations. The course's leaderboard eval (via `hf jobs run` + lighteval) is part of each unit's final-project submission, which I haven't tackled yet for either U1 or U2.
- **Summarization-specific preference data.** The DPO step used a general preference mix, so the gains are in response quality and formatting rather than summarization quality per se. The on-domain dataset I wanted (`openai/summarize_from_feedback`) is gated behind a legacy loading script that current `datasets` refuses to run.
- **A serious chart-reading model.** The U3 adapter learned the chart-answer format, not the chart-reading skill (vision encoder is frozen). Unfreezing the SigLIP encoder, or adding a connector-tuning phase, with more epochs and higher LoRA rank, would land a meaningful chart specialist. Out of scope for the current portfolio framing; would be a fun follow-on.
- **Sample efficiency exploration.** Higher LoRA rank, more epochs, larger subsample. I haven't ablated.

---

## Acknowledgements

- The [Hugging Face Smol Fine-Tuning Language Models course](https://huggingface.co/learn/smol-course/unit0/1) team for the curriculum.
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) and [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) by HuggingFaceTB.
- [TRL](https://github.com/huggingface/trl) for `SFTTrainer`, `DPOTrainer`, and the CLI.

---

## License

Apache 2.0. See `LICENSE`.
