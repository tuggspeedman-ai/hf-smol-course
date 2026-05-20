# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.4,<2.11",
#     "transformers>=5.6",
#     "trl>=1.2",
#     "peft>=0.19",
#     "datasets>=3.0",
#     "accelerate>=1.0",
#     "huggingface_hub>=1.0",
#     "trackio",
#     "numpy",
# ]
# ///
# ^^^ PEP 723 inline deps — read by `hf jobs uv run` to provision the cloud env.
# `torch<2.11` is critical: 2.11+ wheels are CUDA-13 by default, HF Jobs hosts run
# CUDA-12, the mismatch silently falls back to CPU. Cell 1 hard-asserts on that.
# (Same gotcha bit us on the U1 scale run; see .claude/memory/gotchas_uv_torch_cuda_hf_jobs.md)

# %% [markdown]
# # Unit 2 — Exercise 2: DPO on the U1 SFT'd adapter (summarize → preference-aligned)
#
# Direct Preference Optimization on top of `tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora`
# (the U1 adapter), using the SmolTalk2 Tulu 3 preference mix (`_no_think` split).
#
# **LoRA strategy (Path D).** We load the U1 adapter as a `PeftModel` and pass it to
# TRL's `DPOTrainer` with NO `peft_config`. TRL detects the existing PEFT adapter and
# clones `"default"` into a frozen `"ref"` adapter (see TRL `dpo_trainer.py:577-585`),
# trains `"default"` further, uses `"ref"` as the reference policy π_ref in the DPO
# objective. The published artifact is one combined SFT+DPO adapter.
#
# WORKFLOW: run with `SMOKE=True` (200 rows, 50 steps, MPS, no Hub push). Verify the
# smoke assertions and the runtime estimate. If acceptable, flip `SMOKE=False` and
# submit via `hf jobs uv run --flavor a100-large --timeout 4h ...`.
#
# DOWNLOADS: Cell 2 pulls the ~600MB preference parquet shard. Cell 6 pulls the
# instruct tokenizer (~30MB), the SmolLM3-3B base weights (~6.2GB), and the U1 SFT
# adapter (~50MB). Nothing downloads before Cell 2.

# %%
# --- Cell 1: setup, device, seed, full config block ---
# Everything tunable lives here. The SMOKE flag drives dataset sizes and step counts
# so the same code path runs for both smoke check and real run.
import json
import os
import time
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

SEED = 42
# SMOKE defaults to True (safe local default). Override for the scale run:
#   local:  SMOKE=false uv run python notebooks/unit2/exercise2_dpo_lora.py
#   cloud:  hf jobs uv run --env SMOKE=false ... notebooks/unit2/exercise2_dpo_lora.py
SMOKE = os.environ.get("SMOKE", "true").strip().lower() not in ("0", "false", "no", "")

# --- models / adapters / data ---
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"  # SmolLM3-3B base weights
TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"  # instruct tokenizer: has the chat template,
#                                            same vocab as the base model
SFT_ADAPTER_REPO = "tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora"  # U1 adapter
DATASET_REPO = "HuggingFaceTB/smoltalk2"
DATASET_FILE = "Preference/llama_3.1_tulu_3_8b_preference_mixture_no_think-00000-of-00003.parquet"

# --- sizes: smoke vs scale ---
SMOKE_ROWS = 200
SCALE_ROWS = 12_000  # of 76,834 (shard 0 only); 1 epoch → ~1,425 steps at eff. batch 8
#                      (11,400 train rows / 8). Sized to ~$6.75 budget on a100-large.
SMOKE_MAX_STEPS = 50
EVAL_FRACTION = 0.05  # held out BEFORE training (data-leakage guard)

# --- DPO training config ---
# TRL 1.2's DPOConfig has ONE length cap (max_length) covering prompt+completion. The
# course's older `max_prompt_length` kwarg is gone — the only supported truncation_mode
# is "keep_start", which keeps the prompt and truncates the tail of chosen/rejected.
# Cell 4 reports what fraction of pairs that affects on this dataset.
MAX_LENGTH = 1024
PER_DEVICE_BATCH = 1  # at bsz=1 the collator still puts 2 sequences in the tensor
#                       (chosen + rejected) — DPO is heavier per-step than SFT
GRAD_ACCUM = 8  # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM = 8 (matches U1)
LEARNING_RATE = 1e-6  # DPOConfig default. Bumped from the course's 5e-7 after the
#                       first smoke showed near-zero policy movement in 50 steps; still
#                       well within the course's stated 5e-7–5e-6 DPO range.
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.1  # DPO benefits from longer warmup than SFT (loss starts near
#                     log(2)=0.693; gradients near zero need careful initial steps)
NUM_EPOCHS = 1
BETA = 0.1  # KL strength: lower → stays closer to reference (SFT'd model). 0.1 is the
#             standard sigmoid-DPO setting; 0.5 would aggressively push toward chosen
LOSS_TYPE = "sigmoid"  # plain DPO (the loss from Rafailov et al. 2023)
OPTIM = "adamw_torch"  # NOT adamw_torch_fused (CUDA-only)
USE_BF16 = True
GRAD_CHECKPOINTING = True
DATALOADER_WORKERS = 0  # keep 0 on macOS
SAVE_TOTAL_LIMIT = 2
REPORT_TO = ["trackio"]  # custom-script path (not TRL CLI) — U1 Ex3 pattern worked
#                          with trackio + PEFT; fall back to [] if parquet crash recurs

# --- Hub publishing (scale run only) ---
HUB_MODEL_ID = "tuggspeedman-ai/SmolLM3-3B-summarize-dpo-lora"
HUB_PRIVATE = True  # publish PRIVATE first; review the before/after gens + metrics, then
#                     flip to public manually. The smoke showed near-zero DPO effect, so
#                     SCALE output quality is unverified — don't expose it publicly yet.

model_dtype = torch.bfloat16 if USE_BF16 else torch.float32

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

set_seed(SEED)

TAG = "smoke" if SMOKE else datetime.now().strftime("%Y%m%d-%H%M")
RUN_NAME = f"dpo-summarize-{TAG}-beta{BETA}-lr{LEARNING_RATE:.0e}"
RUN_DIR = Path("runs") / RUN_NAME

print(f"device={device}  dtype={model_dtype}  mode={'SMOKE' if SMOKE else 'SCALE'}")
print(f"run_name={RUN_NAME}")
# Fail fast on CPU — same lesson as U1: a CUDA-mismatched torch wheel can silently
# fall back to CPU on HF Jobs and burn cloud compute. Be loud, not a "warning".
assert device != "cpu", (
    "No GPU accelerator detected. On HF Jobs this likely means the installed torch "
    "wheel is built for a different CUDA version than the host driver "
    "(torch>=2.11 is CUDA-13 by default; HF Jobs hosts run CUDA-12). "
    "Check the PEP 723 deps block pins torch<2.11. Locally on Mac you should see device=mps."
)


def generate_response(model, tokenizer, prompt_messages, kwargs, max_new_tokens=256):
    """Greedy-decode an assistant turn from a prompt (list of {role,content}).

    Used for before/after demos. `kwargs` is the row's chat_template_kwargs (carries
    enable_thinking=False) so the prompt is formatted exactly as in training.
    skip_special_tokens=False so chat formatting is visible in the diff.
    """
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **(kwargs or {}),
    ).to(model.device)
    was_training = model.training
    model.eval()
    model.config.use_cache = True  # grad checkpointing disables this during training
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    if was_training:
        model.train()
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


# %%
# --- Cell 2: resolve + download the ONE dataset file ---
# Same gotcha as U1: don't `load_dataset(repo, "Preference", split="X")` — datasets
# resolves the whole config (many GB) before isolating. hf_hub_download the exact
# parquet file. The _no_think variant has 3 shards; we use shard 0 only (76,834 rows
# is more than 6× our SCALE_ROWS=12,000 target — no reason to download the other two).
repo_files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
assert DATASET_FILE in repo_files, (
    f"{DATASET_FILE} not in {DATASET_REPO}. Preference/ files:\n"
    + "\n".join(f for f in repo_files if f.startswith("Preference/"))
)
parquet_path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
print(f"downloaded: {parquet_path}")

# %%
# --- Cell 3: load + schema-validate + reformat for DPO ---
# Schema validation at a system boundary (loaded dataset). The Tulu 3 preference shard
# arrives with columns: prompt:str, chosen:list[{role,content}], rejected:list[{role,
# content}], chat_template_kwargs:dict, source:str. TRL's DPOTrainer needs prompt to
# be either str (standard) or list[message] (conversational) — mixed (str prompt +
# list chosen/rejected) breaks the `_tokenize` path. So we reformat to fully
# conversational: prompt = all turns before the final assistant turn; chosen/rejected
# = the final assistant turn only (the diverging completion).
full_ds = load_dataset("parquet", data_files=parquet_path, split="train")
print(f"{full_ds.num_rows:,} rows  |  columns: {full_ds.column_names}")

EXPECTED_COLS = {"prompt", "chosen", "rejected", "chat_template_kwargs", "source"}
missing = EXPECTED_COLS - set(full_ds.column_names)
assert not missing, f"dataset missing expected columns: {missing}"

# Sanity check: chat_template_kwargs is uniformly /no_think with no tools.
sample_n = min(1000, full_ds.num_rows)
sample_kwargs = full_ds.select(range(sample_n))["chat_template_kwargs"]
enable_vals = {repr((k or {}).get("enable_thinking")) for k in sample_kwargs}
has_tools = any((k or {}).get("xml_tools") or (k or {}).get("python_tools") for k in sample_kwargs)
assert enable_vals == {"False"}, f"expected uniform enable_thinking=False, got {enable_vals}"
assert not has_tools, "unexpected tool-calling rows in a _no_think preference split"
print(f"chat_template_kwargs check: enable_thinking={enable_vals} no_tools={not has_tools}  (ok)")


def reformat_for_dpo(example):
    """Convert (prompt:str, chosen:list, rejected:list) → conversational triple.

    chosen and rejected both start with the user turn(s) (matching `prompt`) and end
    with the diverging assistant turn. We split: prompt = chosen[:-1] (everything
    before the final assistant turn — generalizes to multi-turn, though this dataset
    is single-turn), and chosen / rejected become single-message lists containing just
    the final assistant turn. NOTE: ~1% of source rows have chosen == rejected (zero
    DPO signal); left in to match the published run rather than filtered.
    """
    chosen, rejected = example["chosen"], example["rejected"]
    assert chosen[-1]["role"] == "assistant", f"chosen final turn not assistant: {chosen[-1]}"
    assert rejected[-1]["role"] == "assistant", f"rejected final turn not assistant: {rejected[-1]}"
    # Verify both sides share the same prompt portion (this is the standard preference-
    # dataset invariant; if it fails we'd be silently training on misaligned pairs).
    assert chosen[:-1] == rejected[:-1], (
        f"chosen/rejected differ before the final assistant turn — pair is misaligned. "
        f"source={example.get('source')!r}"
    )
    return {
        "prompt": chosen[:-1],
        "chosen": [chosen[-1]],
        "rejected": [rejected[-1]],
        "chat_template_kwargs": example["chat_template_kwargs"],
    }


# Map (single-threaded — dataset_num_proc>1 can struggle with the assertion-bearing fn).
formatted_ds = full_ds.map(
    reformat_for_dpo,
    remove_columns=[c for c in full_ds.column_names if c not in EXPECTED_COLS]
    + ["prompt"],  # drop the original str prompt; reformat returns a list[message]
    desc="reformat for DPO",
)
print(f"reformatted columns: {formatted_ds.column_names}")
row0 = formatted_ds[0]
print(f"\nsample row 0 (source={full_ds[0]['source']!r}):")
print(f"  prompt ({len(row0['prompt'])} turn(s)):")
for m in row0["prompt"]:
    s = m["content"]
    print(f"    {m['role']}: {s[:160]!r}{' ...[trunc]' if len(s) > 160 else ''}")
print(f"  chosen:   {row0['chosen'][0]['content'][:200]!r}")
print(f"  rejected: {row0['rejected'][0]['content'][:200]!r}")

# %%
# --- Cell 4: token-length analysis -> validate MAX_LENGTH ---
# Done with ONLY the tokenizer (~30MB) BEFORE the ~6.2GB base download. We template
# prompt+chosen and prompt+rejected the same way DPOTrainer's _prepare_dataset will
# (per-row chat_template_kwargs), measure max(prompt+chosen, prompt+rejected) length
# per row — that's the actual ceiling because the collator pads both to the longest.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

len_sample_n = min(2000, formatted_ds.num_rows)
prompt_lens, pair_lens = [], []  # pair_lens = max(prompt+chosen, prompt+rejected) per row
for row in formatted_ds.select(range(len_sample_n)):
    kwargs = row["chat_template_kwargs"] or {}
    p_ids = tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True, **kwargs
    )["input_ids"]
    pc_ids = tokenizer.apply_chat_template(
        row["prompt"] + row["chosen"], tokenize=True, **kwargs
    )["input_ids"]
    pr_ids = tokenizer.apply_chat_template(
        row["prompt"] + row["rejected"], tokenize=True, **kwargs
    )["input_ids"]
    prompt_lens.append(len(p_ids))
    pair_lens.append(max(len(pc_ids), len(pr_ids)))
prompt_lens = np.array(prompt_lens)
pair_lens = np.array(pair_lens)

p_pct = {f"p{p}": int(np.percentile(prompt_lens, p)) for p in (50, 90, 95, 99)}
pair_pct = {f"p{p}": int(np.percentile(pair_lens, p)) for p in (50, 90, 95, 99)}
truncated_pct = float((pair_lens > MAX_LENGTH).mean()) * 100

print(f"prompt token lengths over {len_sample_n} rows: {p_pct} max={prompt_lens.max()}")
print(f"prompt+completion token lengths     : {pair_pct} max={pair_lens.max()}")
print(f"MAX_LENGTH={MAX_LENGTH} would truncate {truncated_pct:.1f}% of pairs in this sample")
# Note: truncation_mode='keep_start' means we keep the prompt portion and drop the
# tail of the chosen/rejected completion. Long-completion examples will train on
# only the start of the response. Bump MAX_LENGTH if this fraction is too high.

RUN_DIR.mkdir(parents=True, exist_ok=True)
length_stats = {
    "sample_rows": len_sample_n,
    "prompt_lens": {
        "min": int(prompt_lens.min()),
        "max": int(prompt_lens.max()),
        "mean": float(prompt_lens.mean()),
        **p_pct,
    },
    "pair_lens": {
        "min": int(pair_lens.min()),
        "max": int(pair_lens.max()),
        "mean": float(pair_lens.mean()),
        **pair_pct,
    },
    "max_length_used": MAX_LENGTH,
    "truncated_pct": truncated_pct,
}
(RUN_DIR / "length_stats.json").write_text(json.dumps(length_stats, indent=2))
print(f"wrote {RUN_DIR / 'length_stats.json'}")

# %%
# --- Cell 5: train/eval split (built BEFORE training — leakage guard) ---
n_rows = SMOKE_ROWS if SMOKE else SCALE_ROWS
n_rows = min(n_rows, formatted_ds.num_rows)
subset = formatted_ds.shuffle(seed=SEED).select(range(n_rows))
split = subset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
print(f"slice={n_rows:,}  ->  train={train_ds.num_rows:,}  eval={eval_ds.num_rows:,}")

# Fixed demo rows for the before/after comparison (same rows for Cells 7 and 10).
demo_rows = [eval_ds[i] for i in range(min(5, eval_ds.num_rows))]

# %%
# --- Cell 6: load base + U1 SFT adapter as a PeftModel ---
# is_trainable=True is critical: PeftModel.from_pretrained defaults to eval mode (the
# adapter params are frozen for inference). Without it, gradients won't flow into the
# adapter and DPO would silently train nothing.
print(f"loading {BASE_MODEL} ...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=model_dtype)
print(f"base params: {base.num_parameters():,}")

print(f"applying {SFT_ADAPTER_REPO} ...")
model = PeftModel.from_pretrained(base, SFT_ADAPTER_REPO, is_trainable=True)
model.to(device)

# Confirm: exactly one adapter named "default" (no "ref" yet — TRL adds it at trainer
# construction in Cell 8). Trainable params should be the LoRA A/B matrices.
adapter_names = list(model.peft_config.keys())
print(f"adapters on model: {adapter_names}")
assert adapter_names == ["default"], f"expected just ['default'], got {adapter_names}"
model.print_trainable_parameters()

emb_rows = model.get_input_embeddings().weight.shape[0]
assert emb_rows >= len(tokenizer), (
    f"base embedding ({emb_rows}) smaller than instruct vocab ({len(tokenizer)})"
)

# %%
# --- Cell 7: BEFORE-training baseline generation (the SFT'd model on demo prompts) ---
# This captures the SFT'd model's behavior — which is also π_ref during DPO. The
# "after" gen in Cell 10 will show how DPO shifted the policy.
before_gens = []
for row in demo_rows:
    out = generate_response(model, tokenizer, row["prompt"], row["chat_template_kwargs"])
    before_gens.append(
        {
            "prompt": row["prompt"],
            "chosen_reference": row["chosen"][0]["content"],
            "rejected_reference": row["rejected"][0]["content"],
            "generation_sft": out,
        }
    )
(RUN_DIR / "generations_before.json").write_text(json.dumps(before_gens, indent=2))
print(f"wrote {RUN_DIR / 'generations_before.json'}  ({len(before_gens)} prompts)")
print("\n--- before[0] (SFT'd model, truncated) ---")
print(before_gens[0]["generation_sft"][:500])

# %%
# --- Cell 8: DPOConfig + DPOTrainer ---
# We pass model (PeftModel with the SFT adapter) and NO peft_config — TRL's trainer
# detects the existing adapter and clones "default" → "ref" at construction time, so
# π_ref is the frozen SFT'd model and π_θ continues training on the SFT adapter.
# (TRL source: trl/trainer/dpo_trainer.py:577-585)
dpo_config = DPOConfig(
    output_dir=str(RUN_DIR),
    seed=SEED,
    # DPO core
    beta=BETA,
    loss_type=LOSS_TYPE,
    max_length=MAX_LENGTH,
    # batch / optimization
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    optim=OPTIM,
    bf16=(USE_BF16 and device == "cuda"),  # CUDA autocast; on MPS the model dtype
    #                                         already drives bf16 compute
    fp16=False,
    gradient_checkpointing=GRAD_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=DATALOADER_WORKERS,
    # duration
    max_steps=SMOKE_MAX_STEPS if SMOKE else -1,
    num_train_epochs=1 if SMOKE else NUM_EPOCHS,
    # logging / eval / saving
    logging_steps=5 if SMOKE else 10,
    eval_strategy="no" if SMOKE else "steps",
    eval_steps=None if SMOKE else 200,
    save_strategy="no" if SMOKE else "steps",
    save_steps=200,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to="none" if SMOKE else REPORT_TO,
    run_name=RUN_NAME,
    remove_unused_columns=False,  # DPO needs chosen/rejected after _prepare_dataset
    disable_tqdm=True,  # tqdm's \r progress bar clobbers the per-step loss log lines
    #                     when stdout is redirected to a file (smoke log + HF Jobs log)
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,  # carries the chat template
    # NO peft_config — model is already PEFT; TRL clones default→ref instead
)

# After construction the model has BOTH adapters; assert it.
adapter_names_after = list(trainer.model.peft_config.keys())
print(f"adapters on model after trainer init: {adapter_names_after}")
assert set(adapter_names_after) == {"default", "ref"}, (
    f"expected default+ref after DPOTrainer init, got {adapter_names_after}"
)

# Resolved config, written immediately (compaction-safe).
resolved_config = {
    "run_name": RUN_NAME,
    "mode": "smoke" if SMOKE else "scale",
    "seed": SEED,
    "base_model": BASE_MODEL,
    "tokenizer_id": TOKENIZER_ID,
    "sft_adapter_repo": SFT_ADAPTER_REPO,
    "dataset_file": DATASET_FILE,
    "device": device,
    "model_dtype": str(model_dtype),
    "beta": BETA,
    "loss_type": LOSS_TYPE,
    "max_length": MAX_LENGTH,
    "per_device_batch": PER_DEVICE_BATCH,
    "grad_accum": GRAD_ACCUM,
    "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "lr_scheduler": LR_SCHEDULER,
    "warmup_ratio": WARMUP_RATIO,
    "num_epochs": 1 if SMOKE else NUM_EPOCHS,
    "max_steps": SMOKE_MAX_STEPS if SMOKE else -1,
    "optim": OPTIM,
    "train_rows": train_ds.num_rows,
    "eval_rows": eval_ds.num_rows,
    "hub_model_id": HUB_MODEL_ID,
    "hub_private": HUB_PRIVATE,
}
(RUN_DIR / "config.json").write_text(json.dumps(resolved_config, indent=2))
print(f"wrote {RUN_DIR / 'config.json'}")

# %%
# --- Cell 9: train (timed) + write metrics ---
# Expected behavior: loss starts near log(2) ≈ 0.693 (π_θ == π_ref at step 0 → DPO
# loss = -log σ(0) = log(2)). rewards/accuracies should trend toward 1.0 — that's the
# fraction of pairs where the policy assigns higher reward to chosen than to rejected.
print(f"training: {'SMOKE' if SMOKE else 'SCALE'}  ->  {RUN_DIR}")
t0 = time.time()
train_result = trainer.train()
wall_s = time.time() - t0

# DPO log_history has loss + reward metrics. Collect all of them per logging step.
log_history = trainer.state.log_history
loss_logs = [e["loss"] for e in log_history if "loss" in e]
acc_logs = [e.get("rewards/accuracies") for e in log_history if "rewards/accuracies" in e]
margin_logs = [e.get("rewards/margins") for e in log_history if "rewards/margins" in e]
assert loss_logs, "no training loss logged — something is wrong with the run"
first_loss, last_loss = loss_logs[0], loss_logs[-1]
n_steps = trainer.state.global_step
sec_per_step = wall_s / max(n_steps, 1)

trend = "DOWN (ok)" if last_loss < first_loss else "NOT decreasing — INVESTIGATE"
print(f"\nwall time: {wall_s:.0f}s  |  steps: {n_steps}  |  {sec_per_step:.2f}s/step")
print(f"loss: {first_loss:.4f} -> {last_loss:.4f}  [{trend}]  (init expected ~0.693 = log 2)")
if acc_logs:
    first_acc, last_acc = acc_logs[0], acc_logs[-1]
    print(f"rewards/accuracies: {first_acc:.3f} -> {last_acc:.3f}  (target: → 1.0)")
if margin_logs:
    first_marg, last_marg = margin_logs[0], margin_logs[-1]
    print(f"rewards/margins:    {first_marg:+.4f} -> {last_marg:+.4f}  (target: positive, growing)")

metrics = {
    "mode": "smoke" if SMOKE else "scale",
    "run_name": RUN_NAME,
    "wall_time_s": round(wall_s, 1),
    "global_steps": n_steps,
    "sec_per_step": round(sec_per_step, 3),
    "steps_per_sec": round(1.0 / sec_per_step, 3),
    "first_loss": first_loss,
    "last_loss": last_loss,
    "loss_trend_ok": last_loss < first_loss,
    "first_accuracy": acc_logs[0] if acc_logs else None,
    "last_accuracy": acc_logs[-1] if acc_logs else None,
    "first_margin": margin_logs[0] if margin_logs else None,
    "last_margin": margin_logs[-1] if margin_logs else None,
    "loss_history": loss_logs,
    "accuracy_history": acc_logs,
    "margin_history": margin_logs,
    "train_runtime_s": train_result.metrics.get("train_runtime"),
}
(RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"wrote {RUN_DIR / 'metrics.json'}")

# %%
# --- Cell 10: drop ref adapter -> save -> push to Hub -> AFTER-gen (best-effort) ---
# CRITICAL ORDERING (from `feedback_cloud_training_robustness`): the irreplaceable
# artifact (adapter) ships to the Hub FIRST. After-gen is best-effort. On 2026-05-18
# we lost a fully-trained adapter because after-gen tipped us past the job timeout.
#
# Delete the "ref" adapter BEFORE save so the artifact contains only the trained
# "default" adapter — otherwise the Hub repo carries a redundant frozen copy of the
# starting SFT'd weights, doubling its size and confusing the load UX.
print(f"adapters before delete: {list(trainer.model.peft_config.keys())}")
trainer.model.delete_adapter("ref")
print(f"adapters after delete:  {list(trainer.model.peft_config.keys())}")
assert list(trainer.model.peft_config.keys()) == ["default"], (
    "delete_adapter('ref') failed; the save would carry the unwanted ref weights"
)

trainer.save_model(str(RUN_DIR))  # DPO-trained LoRA adapter -> RUN_DIR/adapter_model.safetensors
print(f"saved adapter to {RUN_DIR}")

if not SMOKE:
    print(f"\npushing {RUN_DIR} -> {HUB_MODEL_ID} (private={HUB_PRIVATE}) ...")
    create_repo(HUB_MODEL_ID, exist_ok=True, private=HUB_PRIVATE)
    HfApi().upload_folder(
        folder_path=str(RUN_DIR),
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message=f"DPO LoRA adapter (on top of SFT) — run {RUN_NAME}",
    )
    print(f"published: https://huggingface.co/{HUB_MODEL_ID}")

# After-gen is best-effort. The model now has only "default" (DPO-trained); generate
# from it and diff against the BEFORE (SFT'd) generations stored in Cell 7.
try:
    after_gens = []
    for row in demo_rows:
        out = generate_response(
            trainer.model, tokenizer, row["prompt"], row["chat_template_kwargs"]
        )
        after_gens.append(
            {
                "prompt": row["prompt"],
                "chosen_reference": row["chosen"][0]["content"],
                "rejected_reference": row["rejected"][0]["content"],
                "generation_dpo": out,
            }
        )
    (RUN_DIR / "generations_after.json").write_text(json.dumps(after_gens, indent=2))
    print(f"wrote {RUN_DIR / 'generations_after.json'}")

    for i, (b, a) in enumerate(zip(before_gens, after_gens, strict=True)):
        prompt_str = b["prompt"][-1]["content"] if b["prompt"] else "(empty)"
        print(f"\n{'=' * 78}\nDEMO {i}  — user prompt:\n  {prompt_str[:300]!r}")
        print(f"\n  CHOSEN (preferred):\n  {b['chosen_reference'][:300]!r}")
        print(f"  REJECTED:\n  {b['rejected_reference'][:300]!r}")
        print(f"\n  BEFORE (SFT'd model):\n  {b['generation_sft'][:300]!r}")
        print(f"  AFTER  (DPO-aligned):\n  {a['generation_dpo'][:300]!r}")

    if not SMOKE:
        HfApi().upload_file(
            path_or_fileobj=str(RUN_DIR / "generations_after.json"),
            path_in_repo="generations_after.json",
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            commit_message=f"add after-training generations for {RUN_NAME}",
        )
        print(f"pushed generations_after.json to {HUB_MODEL_ID}")
except Exception as e:
    print(f"\nWARN: after-generation step failed (best-effort): {type(e).__name__}: {e}")
    print("The adapter is already safe on the Hub. Skipping after-gen.")

# %%
# --- Cell 11: runtime gate / run summary ---
# For a SMOKE run: extrapolate the scale-run wall time so the >30 min spend can be
# approved before flipping SMOKE=False. For a SCALE run: just report what happened.
if SMOKE:
    eff_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    scale_train_rows = int(min(SCALE_ROWS, formatted_ds.num_rows) * (1 - EVAL_FRACTION))
    scale_steps = ceil(scale_train_rows / eff_batch) * NUM_EPOCHS
    est_min = scale_steps * sec_per_step / 60
    print(f"smoke complete: {sec_per_step:.2f}s/step over {n_steps} steps")
    print(f"\nSCALE-RUN STEPS: {scale_steps:,} (1 epoch over {scale_train_rows:,} rows)")
    # NOTE: do NOT extrapolate cloud wall-time from this smoke's step time — the smoke
    # runs on MPS, the SCALE run on an A100. Same-hardware extrapolation only:
    print(
        f"  if this same hardware ({device}) ran SCALE: {scale_steps:,} x {sec_per_step:.1f}s "
        f"~= {est_min / 60:.1f}h  (irrelevant for the a100-large cloud run)"
    )
    print("  a100-large estimate: ~6 s/step -> ~2.5h, ~$6.75 (see PLAN.md derivation)")
    print(f"  loss trajectory in smoke: {first_loss:.4f} -> {last_loss:.4f}")
    if acc_logs:
        print(f"  rewards/accuracies in smoke: {acc_logs[0]:.3f} -> {acc_logs[-1]:.3f}")
    print("\n  before flipping SMOKE=False:")
    print(f"  - MAX_LENGTH={MAX_LENGTH} truncates {length_stats['truncated_pct']:.1f}% of pairs;")
    print("    consider bumping to 2048 if you want better coverage on long pairs")
    print("  - confirm loss starts near 0.693 (log 2) and trends down")
    print("  - confirm rewards/accuracies climbs above 0.5 by end of smoke")
    if est_min > 30:
        print(f"  - the estimate is >30 min ({est_min:.0f} min) — get explicit approval first")
    print("  - then set SMOKE=False and submit to HF Jobs")
else:
    print(f"SCALE run complete: {RUN_NAME}")
    print(f"  wall time: {wall_s / 60:.1f} min  |  loss {first_loss:.4f} -> {last_loss:.4f}")
    if acc_logs:
        print(f"  rewards/accuracies: {acc_logs[0]:.3f} -> {acc_logs[-1]:.3f}")
    print(f"  artifacts in {RUN_DIR}/  (config, metrics, length_stats, generations, adapter)")
    print(f"  published: https://huggingface.co/{HUB_MODEL_ID}")
    print("  next: review trackio dashboard, before/after generations, model card")

# %%
