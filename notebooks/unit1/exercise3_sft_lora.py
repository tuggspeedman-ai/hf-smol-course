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
# Locally a no-op (uv ignores it when run as `uv run python <file>`); but on HF Jobs
# this is how the runtime knows what to install.
# `torch<2.11` is the critical pin: torch 2.11 defaults to CUDA-13-built wheels,
# but HF Jobs A100 hosts run a CUDA-12 driver — CUDA-13 torch silently falls back to
# CPU (torch.cuda.is_available() == False). Pinning <2.11 keeps us on CUDA-12 wheels.
# (Lost ~$5 to this on 2026-05-18 before catching it; see Cell 1 cuda-required check.)

# %% [markdown]
# # Unit 1 — Exercise 3: Fine-Tuning SmolLM3-3B-Base with LoRA (summarization)
#
# The first real training run. Exercises 1-2 were CPU-only (chat template, dataset
# prep); here we LoRA-fine-tune **SmolLM3-3B-Base** with TRL's `SFTTrainer` on the
# `smol_summarize_no_think` split of SmolTalk2. This is the start of the portfolio
# arc: SFT here (U1) -> DPO (U2) -> publish.
#
# Run cell-by-cell in VSCode (the `# %%` markers render as cells) or in stages with
# `uv run python notebooks/unit1/exercise3_sft_lora.py`.
#
# WORKFLOW: run once with `SMOKE = True` (50 steps, 200 rows, ~minutes). Check the
# smoke assertions and the runtime estimate printed by the last cell. Then, only if
# the estimate is acceptable, flip `SMOKE = False` and re-run for the real (scale) run.
#
# DOWNLOADS: Cell 2 pulls the ~229MB summarize parquet, Cell 4 the ~30MB instruct
# tokenizer, Cell 6 the ~6.2GB SmolLM3-3B-Base weights. Nothing downloads before Cell 2.

# %%
# --- Cell 1: setup, device, seed, and the full config block ---
# Everything tunable lives here. The SMOKE flag drives dataset sizes and step counts
# so the exact same code path runs for both the smoke check and the real run.
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
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

SEED = 42
# SMOKE defaults to True (safe local default). Override for the scale run via env var:
#   local:  SMOKE=false uv run python notebooks/unit1/exercise3_sft_lora.py
#   cloud:  hf jobs uv run --env SMOKE=false ... notebooks/unit1/exercise3_sft_lora.py
SMOKE = os.environ.get("SMOKE", "true").strip().lower() not in ("0", "false", "no", "")

# --- models / data ---
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"  # we fine-tune the BASE model
TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"  # instruct tokenizer: has the chat template,
#                                            same vocab as the base model
DATASET_REPO = "HuggingFaceTB/smoltalk2"
DATASET_FILE = "SFT/smoltalk_smollm3_smol_summarize_no_think-00000-of-00001.parquet"

# --- LoRA (PEFT 0.19 has no smollm3 target-module preset, so we pass them explicitly;
#     SmolLM3 is Llama-style: 4 attention + 3 MLP projections) ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- sizes: smoke vs scale ---
SMOKE_ROWS = 200
SCALE_ROWS = 12_000  # of 96,061; sized to fit the cloud-run budget at 3.66 s/step on
#                      a100-large (~87 min, ~$3.60). 12k → 1,425 optimizer steps,
#                      28× the smoke — well past where the loss curve stabilizes.
SMOKE_MAX_STEPS = 50
EVAL_FRACTION = 0.05  # held out BEFORE training (data-leakage guard)

# --- training config ---
MAX_LENGTH = 2304  # from Cell 4 on the full split: p99=2063 -> 2304 (covers ~99% of
#                    full sequences; max=2517). Re-run Cell 4 if the split changes.
PER_DEVICE_BATCH = 1  # conservative MPS starting point
GRAD_ACCUM = 8  # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM = 8
LEARNING_RATE = 2e-4  # standard LoRA LR (the course's 5e-5 is a full-FT LR)
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
NUM_EPOCHS = 1  # scale run: 1 epoch over SCALE_ROWS
OPTIM = "adamw_torch"  # NOT adamw_torch_fused (TRL's default) — fused is CUDA-only
USE_BF16 = True  # load the model in bf16 (memory + speed win on MPS); fp32 fallback below
GRAD_CHECKPOINTING = True
DATALOADER_WORKERS = 0  # keep at 0 on macOS
ASSISTANT_ONLY_LOSS = True  # use the chat template's {% generation %} mask -> loss only
#                             on assistant tokens (the "completion-only" loss from Ex1)
PACKING = False  # naive-first; no flash-attn on MPS, and packing risks cross-contamination
SAVE_TOTAL_LIMIT = 2
REPORT_TO = ["trackio"]  # HF-native tracker; used for the scale run only

# --- Hub publishing (scale run only) ---
HUB_MODEL_ID = "tuggspeedman-ai/SmolLM3-3B-summarize-sft-lora"  # adapter + tokenizer +
#                            run artifacts get pushed here at end of Cell 10 when SMOKE=False

model_dtype = torch.bfloat16 if USE_BF16 else torch.float32

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

set_seed(SEED)

# (Tried a CUDA override here on 2026-05-18: batch=4 + grad_ckpt=False on a100-large
# OOM'd at backward — attention scores at seq=2304 don't fit even on 80GB without
# checkpointing. Reverted; verified config above works at 3.66 s/step on A100.)

# Run naming: smoke runs all share one dir; scale runs are timestamped.
TAG = "smoke" if SMOKE else datetime.now().strftime("%Y%m%d-%H%M")
RUN_NAME = f"sft-summarize-{TAG}-r{LORA_R}-lr{LEARNING_RATE:.0e}"
RUN_DIR = Path("runs") / RUN_NAME

print(f"device={device}  dtype={model_dtype}  mode={'SMOKE' if SMOKE else 'SCALE'}")
print(f"run_name={RUN_NAME}")
# Fail fast on CPU. Training a 3B model on CPU is impractical no matter the setting —
# previously a CUDA-mismatched torch wheel silently fell back to CPU on HF Jobs and
# we burned ~$5 of A100 time before noticing. Make this loud, not a "warning".
assert device != "cpu", (
    "No GPU accelerator detected. On HF Jobs this likely means the installed torch "
    "wheel is built for a different CUDA version than the host driver "
    "(torch>=2.11 is CUDA-13 by default; HF Jobs hosts run CUDA-12). "
    "Check the PEP 723 deps block at the top of this file pins torch<2.11. "
    "Locally on Mac you should see device=mps."
)


def generate_response(model, tokenizer, messages, kwargs, max_new_tokens=256):
    """Greedy-decode the assistant turn for a conversation prompt.

    `messages` is a full conversation row; we drop the final assistant turn and let
    the model regenerate it. `kwargs` is the row's chat_template_kwargs (carries
    enable_thinking=False etc.) so the prompt is formatted exactly as in training.
    skip_special_tokens=False so the chat formatting is visible in before/after.
    """
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
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
    model.config.use_cache = True  # gradient checkpointing disables this during training
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -> reproducible before/after comparison
            pad_token_id=tokenizer.pad_token_id,
        )
    if was_training:
        model.train()
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


# %%
# --- Cell 2: resolve + download the ONE dataset file ---
# GOTCHA (see .claude/memory/gotchas_smoltalk2_download.md): load_dataset(repo, "SFT",
# split="X") resolves the whole 66GB config before isolating the split. The safe path
# is hf_hub_download of the exact parquet file (each split == one file).
repo_files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
assert DATASET_FILE in repo_files, (
    f"{DATASET_FILE} not found in {DATASET_REPO}. SFT/ files containing 'summarize':\n"
    + "\n".join(f for f in repo_files if "summarize" in f.lower())
)
parquet_path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
print(f"downloaded: {parquet_path}")

# %%
# --- Cell 3: load + schema-validate the split ---
# Schema validation at a system boundary (a loaded dataset): confirm the columns
# SFTTrainer needs are present, and confirm our "all /no_think" assumption holds —
# that assumption is WHY we can hand the raw dataset to SFTTrainer (see Cell 8).
full_ds = load_dataset("parquet", data_files=parquet_path, split="train")
print(f"{full_ds.num_rows:,} rows  |  columns: {full_ds.column_names}")

EXPECTED_COLS = {"messages", "chat_template_kwargs", "source"}
missing = EXPECTED_COLS - set(full_ds.column_names)
assert not missing, f"dataset missing expected columns: {missing}"

# Inspect chat_template_kwargs across a sample — every row in a *_no_think split should
# carry enable_thinking=False and no tools. If that holds, SFTTrainer's per-row
# templating (which passes chat_template_kwargs through) gives correct /no_think format.
sample_n = min(1000, full_ds.num_rows)
sample_kwargs = full_ds.select(range(sample_n))["chat_template_kwargs"]
enable_vals = {repr((k or {}).get("enable_thinking")) for k in sample_kwargs}
has_tools = any((k or {}).get("xml_tools") or (k or {}).get("python_tools") for k in sample_kwargs)
print(f"chat_template_kwargs over {sample_n} rows: enable_thinking values = {enable_vals}")
print(f"any tool-calling rows: {has_tools}")
assert enable_vals == {"False"}, f"expected uniform enable_thinking=False, got {enable_vals}"
assert not has_tools, "unexpected tool-calling rows in a summarize split"

row0 = full_ds[0]
print(f"\nsample row — source={row0['source']!r}, kwargs={row0['chat_template_kwargs']!r}")
for i, m in enumerate(row0["messages"]):
    preview = m["content"] if len(m["content"]) <= 220 else m["content"][:220] + " ...[trunc]"
    print(f"  [{i}] {m['role']}: {preview!r}")

# %%
# --- Cell 4: token-length analysis -> finalize MAX_LENGTH ---
# Done with ONLY the tokenizer (~30MB) BEFORE the ~6.2GB model download — know the
# data before committing to the big pull. We template exactly as SFTTrainer will
# (full conversation, no generation prompt, per-row kwargs) and measure token lengths.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

len_sample_n = min(2000, full_ds.num_rows)
lengths = []
for row in full_ds.select(range(len_sample_n)):
    # transformers 5.6: apply_chat_template(tokenize=True) returns a BatchEncoding,
    # not a bare token list — take ["input_ids"] for the actual ids.
    enc = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, **(row["chat_template_kwargs"] or {})
    )
    lengths.append(len(enc["input_ids"]))
lengths = np.array(lengths)

# Recommend off p99, not p95: truncation clips the END of the sequence — which is the
# assistant summary, i.e. the training label. With PER_DEVICE_BATCH=1 and no packing,
# max_length is only a truncation ceiling (no padding cost), so size it generously.
pct = {f"p{p}": int(np.percentile(lengths, p)) for p in (50, 90, 95, 99)}
recommended = int(min(4096, np.ceil(np.percentile(lengths, 99) / 256) * 256))
print(
    f"token lengths over {len_sample_n} rows: "
    f"min={lengths.min()} {pct} max={lengths.max()} mean={lengths.mean():.0f}"
)
print(f"recommended MAX_LENGTH (p99 rounded up to /256, capped 4096): {recommended}")
print(
    f"current MAX_LENGTH constant: {MAX_LENGTH}"
    f"{'  <-- update Cell 1 and re-run' if MAX_LENGTH != recommended else '  (ok)'}"
)

RUN_DIR.mkdir(parents=True, exist_ok=True)
length_stats = {
    "sample_rows": len_sample_n,
    "min": int(lengths.min()),
    "max": int(lengths.max()),
    "mean": float(lengths.mean()),
    **pct,
    "recommended_max_length": recommended,
    "max_length_used": MAX_LENGTH,
}
(RUN_DIR / "length_stats.json").write_text(json.dumps(length_stats, indent=2))
print(f"wrote {RUN_DIR / 'length_stats.json'}")

# %%
# --- Cell 5: train/eval split (built BEFORE training — leakage guard) ---
# Shuffle with the fixed seed, take the smoke/scale slice, then hold out EVAL_FRACTION.
n_rows = SMOKE_ROWS if SMOKE else SCALE_ROWS
n_rows = min(n_rows, full_ds.num_rows)
subset = full_ds.shuffle(seed=SEED).select(range(n_rows))
split = subset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
print(f"slice={n_rows:,}  ->  train={train_ds.num_rows:,}  eval={eval_ds.num_rows:,}")

# Fixed demo rows for the before/after comparison (same rows used in Cells 7 and 10).
demo_rows = [eval_ds[i] for i in range(min(5, eval_ds.num_rows))]

# %%
# --- Cell 6: load the base model (~6.2GB download on first run) ---
# SmolLM3 is native in transformers 5.6 — no trust_remote_code needed. We load in
# bf16 for the memory/speed win; if training NaNs on MPS, set USE_BF16=False (fp32).
print(f"loading {BASE_MODEL} ...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=model_dtype)
model.to(device)
print(f"params: {model.num_parameters():,}")

# The base model never saw the chat template, but base & instruct share the vocab.
# Assert the embedding matrix can index every token the instruct tokenizer produces.
emb_rows = model.get_input_embeddings().weight.shape[0]
print(f"embedding rows: {emb_rows:,}  |  tokenizer vocab: {len(tokenizer):,}")
assert emb_rows >= len(tokenizer), (
    f"base embedding ({emb_rows}) smaller than instruct vocab ({len(tokenizer)}) — "
    "would need model.resize_token_embeddings(len(tokenizer))"
)

# %%
# --- Cell 7: BEFORE-training baseline generation ---
# Generate on the held-out demo docs with the *pure* base model (still un-LoRA'd here,
# un-trained). Expect rambling / raw continuation, not a clean chat-formatted summary.
before_gens = []
for row in demo_rows:
    out = generate_response(model, tokenizer, row["messages"], row["chat_template_kwargs"])
    before_gens.append(
        {
            "prompt": [m for m in row["messages"] if m["role"] != "assistant"],
            "reference_summary": row["messages"][-1]["content"],
            "generation": out,
        }
    )
(RUN_DIR / "generations_before.json").write_text(json.dumps(before_gens, indent=2))
print(f"wrote {RUN_DIR / 'generations_before.json'}  ({len(before_gens)} docs)")
print("\n--- before[0] (truncated) ---")
print(before_gens[0]["generation"][:500])

# %%
# --- Cell 8: LoRA config + SFTConfig + SFTTrainer ---
# KEY: we pass the RAW dataset (messages + chat_template_kwargs columns), NOT a
# pre-formatted `text` column. TRL 1.2's SFTTrainer templates each row internally and
# passes that row's chat_template_kwargs through apply_chat_template — so /no_think
# formatting is automatic. With assistant_only_loss=True it also requests the
# {% generation %} mask, giving completion-only loss. processing_class MUST be the
# instruct tokenizer (the base tokenizer has no chat template).
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir=str(RUN_DIR),
    seed=SEED,
    # data / sequence
    max_length=MAX_LENGTH,
    packing=PACKING,
    assistant_only_loss=ASSISTANT_ONLY_LOSS,
    # batch / optimization
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    optim=OPTIM,
    bf16=(USE_BF16 and device == "cuda"),  # autocast on CUDA only; on MPS the model
    #                                         dtype already drives bf16 compute
    fp16=False,
    gradient_checkpointing=GRAD_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=DATALOADER_WORKERS,
    # duration: max_steps for smoke, full epochs for scale
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
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,  # instruct tokenizer — carries the chat template
    peft_config=lora_config,
)
trainer.model.print_trainable_parameters()

# Resolved config, written immediately (compaction-safe).
resolved_config = {
    "run_name": RUN_NAME,
    "mode": "smoke" if SMOKE else "scale",
    "seed": SEED,
    "base_model": BASE_MODEL,
    "tokenizer_id": TOKENIZER_ID,
    "dataset_file": DATASET_FILE,
    "device": device,
    "model_dtype": str(model_dtype),
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
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
    "assistant_only_loss": ASSISTANT_ONLY_LOSS,
    "packing": PACKING,
    "train_rows": train_ds.num_rows,
    "eval_rows": eval_ds.num_rows,
}
(RUN_DIR / "config.json").write_text(json.dumps(resolved_config, indent=2))
print(f"wrote {RUN_DIR / 'config.json'}")

# %%
# --- Cell 9: train (timed) + write metrics ---
print(f"training: {'SMOKE' if SMOKE else 'SCALE'}  ->  {RUN_DIR}")
t0 = time.time()
train_result = trainer.train()
wall_s = time.time() - t0

loss_logs = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
assert loss_logs, "no training loss was logged — something is wrong with the run"
first_loss, last_loss = loss_logs[0], loss_logs[-1]
n_steps = trainer.state.global_step
sec_per_step = wall_s / max(n_steps, 1)

trend = "DOWN (ok)" if last_loss < first_loss else "NOT decreasing — INVESTIGATE"
print(f"\nwall time: {wall_s:.0f}s  |  steps: {n_steps}  |  {sec_per_step:.2f}s/step")
print(f"loss: {first_loss:.4f} -> {last_loss:.4f}  [{trend}]")

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
    "loss_history": loss_logs,
    "train_runtime_s": train_result.metrics.get("train_runtime"),
}
(RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"wrote {RUN_DIR / 'metrics.json'}")

# %%
# --- Cell 10: save adapter + push to Hub + AFTER-training generation (best-effort) ---
# CRITICAL ORDERING: save the irreplaceable artifact (LoRA adapter) and push it to the
# Hub IMMEDIATELY, BEFORE the after-generation. On 2026-05-18 we lost a fully-trained
# adapter because the after-gen loop pushed us past the job timeout — the cloud
# filesystem was discarded with the canceled container. After-gen is "nice to have"
# (gravy for the model card); the adapter is "the whole point of the run." Push first.
trainer.save_model(str(RUN_DIR))  # LoRA adapter -> RUN_DIR/adapter_model.safetensors
print(f"saved adapter to {RUN_DIR}")

if not SMOKE:
    print(f"\npushing {RUN_DIR} -> https://huggingface.co/{HUB_MODEL_ID} ...")
    create_repo(HUB_MODEL_ID, exist_ok=True, private=False)
    HfApi().upload_folder(
        folder_path=str(RUN_DIR),
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message=f"SFT LoRA adapter — run {RUN_NAME}",
    )
    print(f"published: https://huggingface.co/{HUB_MODEL_ID}")

# After-gen is now best-effort. If anything fails (timeout, NaN, etc.) the adapter
# is already safe on the Hub. If it succeeds, push the JSON as a follow-up commit.
try:
    after_gens = []
    for row in demo_rows:
        out = generate_response(
            trainer.model, tokenizer, row["messages"], row["chat_template_kwargs"]
        )
        after_gens.append(
            {
                "prompt": [m for m in row["messages"] if m["role"] != "assistant"],
                "reference_summary": row["messages"][-1]["content"],
                "generation": out,
            }
        )
    (RUN_DIR / "generations_after.json").write_text(json.dumps(after_gens, indent=2))
    print(f"wrote {RUN_DIR / 'generations_after.json'}")

    for i, (b, a) in enumerate(zip(before_gens, after_gens, strict=True)):
        print(f"\n{'=' * 78}\nDEMO {i}  — reference summary:\n  {b['reference_summary'][:300]!r}")
        print(f"\n  BEFORE (base model):\n  {b['generation'][:300]!r}")
        print(f"\n  AFTER  (LoRA fine-tuned):\n  {a['generation'][:300]!r}")

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
# For a SMOKE run: extrapolate the scale-run wall time so the >30min spend can be
# approved before flipping SMOKE=False. For a SCALE run: just report what happened.
if SMOKE:
    eff_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    scale_train_rows = int(min(SCALE_ROWS, full_ds.num_rows) * (1 - EVAL_FRACTION))
    scale_steps = ceil(scale_train_rows / eff_batch) * NUM_EPOCHS
    est_min = scale_steps * sec_per_step / 60
    print(f"smoke complete: {sec_per_step:.2f}s/step over {n_steps} steps")
    print(
        f"\nSCALE-RUN ESTIMATE: {scale_steps:,} steps x {sec_per_step:.2f}s "
        f"~= {est_min:.0f} min ({est_min / 60:.1f}h)"
    )
    print("  before flipping SMOKE=False:")
    print("  - update MAX_LENGTH in Cell 1 from the Cell 4 recommendation if needed")
    if est_min > 30:
        print(f"  - the estimate is >30 min ({est_min:.0f} min) — get explicit approval first")
    print("  - then set SMOKE=False and re-run the whole file")
else:
    print(f"SCALE run complete: {RUN_NAME}")
    print(f"  wall time: {wall_s / 60:.1f} min  |  loss {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  artifacts in {RUN_DIR}/  (config, metrics, length_stats, generations, adapter)")
    print("  next: review the trackio dashboard and the before/after generations")

# %%
