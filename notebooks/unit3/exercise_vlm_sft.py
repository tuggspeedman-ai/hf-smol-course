# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.4,<2.11",
#     "torchvision>=0.19,<0.26",
#     "transformers>=5.6",
#     "trl>=1.2",
#     "peft>=0.19",
#     "datasets>=3.0",
#     "accelerate>=1.0",
#     "huggingface_hub>=1.0",
#     "trackio",
#     "numpy",
#     "Pillow>=10.0",
#     "num2words==0.5.14",
# ]
# ///
# ^^^ PEP 723 inline deps — read by `hf jobs uv run` to provision the cloud env.
# `torch<2.11` is critical: 2.11+ wheels are CUDA-13 by default; HF Jobs hosts run
# CUDA-12; mismatch silently falls back to CPU. Cell 1 hard-asserts on that.
# `torchvision<0.26` keeps it compatible with the torch<2.11 pin.
# `torchvision` and `num2words` are both hard deps of SmolVLM's processor — without
# them, `AutoProcessor.from_pretrained` raises before any training starts.
# (See .claude/memory/gotchas_uv_torch_cuda_hf_jobs.md.)

# %% [markdown]
# # Unit 3 — VLM SFT: SmolVLM2-2.2B-Instruct on ChartQA
#
# Supervised fine-tuning of a VLM. New mechanics vs U1/U2: the **processor** (image +
# text, not just a tokenizer), the `{"type":"image"}/{"type":"text"}` chat content
# format, the `images`+`messages` dataset schema, and `AutoModelForImageTextToText`.
#
# **LoRA strategy.** GOTCHA: `peft` matches `target_modules` by suffix across the
# *whole* model, and SmolVLM's SigLIP encoder reuses `q_proj`/`v_proj` names. The
# textbook `["q_proj","v_proj"]` recipe therefore silently LoRA-adapts the vision
# tower too — and TRL's SFTTrainer freezes nothing automatically. We scope LoRA via a
# regex to `model.text_model.*` only (q/k/v/o + MLP). The vision encoder is left
# frozen as a base-model component (PEFT freezes all non-target base params).
#
# **max_length trap.** Set `max_length=None` in Python. The CLI `--max_length -1`
# does NOT disable truncation in TRL 1.2 (HfArgumentParser collapses int|None → int);
# `-1` stays `-1`, truncation stays on, image tokens get chopped.
#
# WORKFLOW: run with `SMOKE=True` (small slice on MPS, no Hub push). Verify smoke
# assertions and the runtime estimate. If acceptable, set `SMOKE=False` and submit
# via `hf jobs uv run --flavor a10g-large --timeout 2h ...`.
#
# DOWNLOADS: Cell 2 pulls the ChartQA `train[:10%]` + `val[:10%]` parquet (~hundreds
# of MB, images embedded). Cell 6 pulls the processor (~10MB), then the 2.2B model
# weights (~4.5GB bf16). Nothing downloads before Cell 2.

# %%
# --- Cell 1: setup, device, seed, full config block ---
# PYTORCH_ENABLE_MPS_FALLBACK must be set BEFORE torch ops so any vision op not
# implemented on MPS (rare but possible in SigLIP) falls back to CPU instead of crashing.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import re
import time
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from trl import SFTConfig, SFTTrainer
from trl.data_utils import prepare_multimodal_messages

SEED = 42
# SMOKE defaults to True (safe local default). Override for the scale run:
#   local:  SMOKE=false uv run python notebooks/unit3/exercise_vlm_sft.py
#   cloud:  hf jobs uv run --env SMOKE=false ... notebooks/unit3/exercise_vlm_sft.py
SMOKE = os.environ.get("SMOKE", "true").strip().lower() not in ("0", "false", "no", "")

# --- model / dataset ---
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATASET_ID = "HuggingFaceM4/ChartQA"
TRAIN_SPLIT = "train[:10%]"
EVAL_SPLIT = "val[:10%]"

# --- sizes: smoke vs scale ---
# ChartQA train has ~28k rows; train[:10%] ≈ 2,830 rows. SCALE uses all of train[:10%];
# SMOKE caps to a tiny subset so MPS finishes in minutes.
SMOKE_ROWS = 64
SMOKE_MAX_STEPS = 20
SMOKE_EVAL_ROWS = 16

# --- LoRA scope ---
# Regex scoped to text_model only (LM attention + MLP). The PEFT injector matches
# this via re.fullmatch against named_modules keys (whole-model). Excludes
# `model.vision_model.*` and `model.connector.*` by construction.
LORA_TARGET_REGEX = (
    r"model\.text_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# --- SFT training config ---
# max_length=None disables truncation — critical for VLM (image tokens must not be
# chopped). The CLI `--max_length -1` does NOT work; only None in Python does.
MAX_LENGTH = None
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8  # effective batch = 8 (matches U1/U2 for cross-unit comparability)
LEARNING_RATE = 1e-4  # course's Ex2 value; reasonable for LoRA on a small VLM
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
NUM_EPOCHS = 1
OPTIM = "adamw_torch"  # NOT adamw_torch_fused (CUDA-only)
USE_BF16 = True
GRAD_CHECKPOINTING = True
DATALOADER_WORKERS = 0  # 0 on macOS
SAVE_TOTAL_LIMIT = 2
REPORT_TO = ["trackio"]

# --- Hub publishing (scale run only) ---
HUB_MODEL_ID = "tuggspeedman-ai/SmolVLM2-2.2B-chartqa-lora"
HUB_PRIVATE = True  # private first, then /review, then flip public

# --- system prompt (course's chart-analyst framing) ---
SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialized in interpreting visual data from "
    "chart images. Your task is to analyze the provided chart image and respond to "
    "queries with concise answers, usually a single word, number, or short phrase. "
    "The charts include a variety of types (e.g., line charts, bar charts) and "
    "contain colors, labels, and text. Focus on delivering accurate, succinct "
    "answers based on the visual information. Avoid additional explanation unless "
    "absolutely necessary."
)

model_dtype = torch.bfloat16 if USE_BF16 else torch.float32

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

set_seed(SEED)

TAG = "smoke" if SMOKE else datetime.now().strftime("%Y%m%d-%H%M")
RUN_NAME = f"vlm-sft-chartqa-{TAG}-r{LORA_R}-lr{LEARNING_RATE:.0e}"
RUN_DIR = Path("runs") / RUN_NAME

print(f"device={device}  dtype={model_dtype}  mode={'SMOKE' if SMOKE else 'SCALE'}")
print(f"run_name={RUN_NAME}")
# Fail fast on CPU — same lesson as U1/U2. A CUDA-mismatched torch wheel can silently
# fall back to CPU on HF Jobs and burn cloud compute.
assert device != "cpu", (
    "No GPU accelerator detected. On HF Jobs this likely means the installed torch "
    "wheel is built for a different CUDA version than the host driver "
    "(torch>=2.11 is CUDA-13 by default; HF Jobs hosts run CUDA-12). "
    "Check the PEP 723 deps block pins torch<2.11. Locally on Mac you should see device=mps."
)


# %%
# --- Cell 2: load ChartQA (scoped slice) + schema check ---
# `load_dataset` with a split slice still downloads the full parquet then slices in-memory;
# for ChartQA's ~28k-row train shard with embedded images this is ~hundreds of MB.
# Acceptable. Larger datasets would warrant `streaming=True` or per-shard `hf_hub_download`.
print(f"loading {DATASET_ID} splits=[{TRAIN_SPLIT!r}, {EVAL_SPLIT!r}] ...")
train_raw, eval_raw = load_dataset(DATASET_ID, split=[TRAIN_SPLIT, EVAL_SPLIT])
print(f"  train={train_raw.num_rows:,}  eval={eval_raw.num_rows:,}  columns={train_raw.column_names}")

EXPECTED_COLS = {"image", "query", "label"}
missing = EXPECTED_COLS - set(train_raw.column_names)
assert not missing, f"ChartQA missing expected columns: {missing}"

# Inspect one row to confirm dtypes + the `label` shape (course noted it's a list).
row0 = train_raw[0]
assert row0["image"] is not None, "row 0 has no image"
assert isinstance(row0["label"], list) and len(row0["label"]) >= 1, (
    f"expected label: list[str], got {type(row0['label']).__name__}: {row0['label']!r}"
)
img0 = row0["image"]
print(
    f"  sample row 0: image={img0.size} mode={img0.mode}  "
    f"query={row0['query'][:80]!r}  label={row0['label'][0]!r}"
)


# %%
# --- Cell 3: format into {images, messages} + show one example ---
# SFTTrainer's VLM collator dispatches on dataset columns:
#   - `messages` (conversational): trains on the WHOLE sequence. `assistant_only_loss`
#     is FORBIDDEN for VLMs (sft_trainer.py:744-748), so there's no way to mask the
#     prompt or image tokens. For ChartQA (1-3 token answers, ~80 image tokens, 200
#     total seq), most of the loss signal comes from "predict the image-token IDs",
#     which is noise — the LM has no business predicting those. First smoke confirmed:
#     starting loss ≈10.33 (≈ uniform over the 49k vocab) and 2/3 after-gens were
#     byte-identical to base after 20 steps.
#   - `prompt`/`completion`: collator builds a `completion_mask` and auto-enables
#     `completion_only_loss=True` (sft_trainer.py:868-871, 508-512). Loss is computed
#     only on the assistant answer tokens — exactly the signal we want.
#
# We use prompt/completion. Both fields are list-of-messages (conversational).
#
# GOTCHA — image placeholder shape: the user content's image block is a BARE
# `{"type": "image"}` (no `image` key). TRL's `prepare_multimodal_messages` counts
# only image blocks WITHOUT an `image` key as placeholders (data_utils.py:98-106) and
# fills them from the `images` column. The course's unit3/4.md example pre-fills with
# `"image": sample["image"]` inside content, which yields 0 placeholders, mismatches
# the 1 image in `images`, and raises against TRL 1.2's collator.
def format_data(sample):
    return {
        "images": [sample["image"]],
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # placeholder; filled from images column
                    {"type": "text", "text": sample["query"]},
                ],
            },
        ],
        "completion": [
            {"role": "assistant", "content": [{"type": "text", "text": sample["label"][0]}]},
        ],
    }


# Map keeps the dataset interface; .map removes the original columns explicitly so the
# collator only sees what it needs. `keep_in_memory=False` since images are heavy.
train_ds_full = train_raw.map(format_data, remove_columns=train_raw.column_names, desc="format train")
eval_ds_full = eval_raw.map(format_data, remove_columns=eval_raw.column_names, desc="format eval")
print(f"formatted columns: {train_ds_full.column_names}")
ex = train_ds_full[0]
print(
    f"  example 0: images=[{ex['images'][0].size}]  "
    f"prompt_roles={[m['role'] for m in ex['prompt']]}  "
    f"completion={ex['completion'][0]['content'][0]['text']!r}"
)


# %%
# --- Cell 4: image-token length analysis (sanity check) ---
# Load the processor early (~10MB) BEFORE the 4.5GB model download so we can measure
# real sequence lengths with image-token expansion. This mirrors the path the collator
# takes: prepare_multimodal_messages → apply_chat_template → processor(images, text).
print(f"loading processor {MODEL_ID} ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Smoke caps image-splitting to keep MPS feasible — 1 frame × 81 tokens instead of
# ≤17 frames × 81 ≈ 1,400 tokens. Scale run uses the default (splitting on, full res).
if SMOKE:
    processor.image_processor.do_image_splitting = False
    print("  SMOKE: do_image_splitting=False  (caps to ~81 image tokens/example)")
else:
    print(f"  SCALE: do_image_splitting={processor.image_processor.do_image_splitting}")

# For length analysis we concat prompt+completion (same total tokens as what the
# prompt/completion collator builds at training time; we just don't split them).
len_sample_n = min(50, train_ds_full.num_rows)
seq_lens, completion_lens = [], []
for i in range(len_sample_n):
    row = train_ds_full[i]
    combined = list(row["prompt"]) + list(row["completion"])
    prepared = prepare_multimodal_messages(combined, images=row["images"])
    text = processor.apply_chat_template(prepared)
    out = processor(
        images=row["images"],
        text=text,
        return_tensors=None,
        padding=False,
        add_special_tokens=False,  # mirrors collator
    )
    seq_lens.append(len(out["input_ids"][0]))
    completion_lens.append(
        len(processor.tokenizer.encode(row["completion"][0]["content"][0]["text"], add_special_tokens=False))
    )
seq_lens = np.array(seq_lens)
completion_lens = np.array(completion_lens)
seq_pct = {f"p{p}": int(np.percentile(seq_lens, p)) for p in (50, 90, 95, 99)}
print(
    f"seq lengths over {len_sample_n} rows: "
    f"min={seq_lens.min()}  max={seq_lens.max()}  mean={seq_lens.mean():.0f}  {seq_pct}"
)
print(
    f"completion-only token lengths: "
    f"min={completion_lens.min()}  max={completion_lens.max()}  "
    f"mean={completion_lens.mean():.1f}  (this is what loss is computed on)"
)

RUN_DIR.mkdir(parents=True, exist_ok=True)
length_stats = {
    "sample_rows": len_sample_n,
    "do_image_splitting": processor.image_processor.do_image_splitting,
    "seq_lens": {
        "min": int(seq_lens.min()),
        "max": int(seq_lens.max()),
        "mean": float(seq_lens.mean()),
        **seq_pct,
    },
    "max_length_used": MAX_LENGTH,
}
(RUN_DIR / "length_stats.json").write_text(json.dumps(length_stats, indent=2))
print(f"wrote {RUN_DIR / 'length_stats.json'}")


# %%
# --- Cell 5: select train/eval subset (sizes depend on SMOKE) ---
# ChartQA already ships train/val splits; no need to re-split. Just take N from each.
if SMOKE:
    train_ds = train_ds_full.shuffle(seed=SEED).select(range(min(SMOKE_ROWS, train_ds_full.num_rows)))
    eval_ds = eval_ds_full.select(range(min(SMOKE_EVAL_ROWS, eval_ds_full.num_rows)))
else:
    train_ds = train_ds_full
    eval_ds = eval_ds_full.select(range(min(200, eval_ds_full.num_rows)))
print(f"train={train_ds.num_rows:,}  eval={eval_ds.num_rows:,}")

# Fixed demo rows for the before/after diff (use eval — model has not seen them).
demo_rows = [eval_ds[i] for i in range(min(3, eval_ds.num_rows))]


# %%
# --- Cell 6: load model, build LoRA config, assert LM-only scope ---
print(f"loading {MODEL_ID} (dtype={model_dtype}) ...")
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=model_dtype)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  loaded: {n_params:,} params")

# Verify expected submodule names (the regex relies on these).
assert hasattr(model.model, "vision_model"), "expected model.model.vision_model"
assert hasattr(model.model, "text_model"), "expected model.model.text_model"
assert hasattr(model.model, "connector"), "expected model.model.connector"

# Sanity-check the regex against actual module names BEFORE handing to PEFT — fail
# fast here, not deep inside trainer construction.
all_keys = [k for k, _ in model.named_modules()]
text_matches = [k for k in all_keys if re.fullmatch(LORA_TARGET_REGEX, k)]
vision_matches = [
    k
    for k in all_keys
    if re.fullmatch(LORA_TARGET_REGEX, k) and k.startswith("model.vision_model.")
]
assert text_matches, f"LoRA regex matched 0 modules; check LORA_TARGET_REGEX={LORA_TARGET_REGEX!r}"
assert not vision_matches, (
    f"LoRA regex unexpectedly matches {len(vision_matches)} vision-encoder modules: "
    f"{vision_matches[:3]}..."
)
print(f"  LoRA regex matches {len(text_matches)} modules; 0 in vision encoder (ok)")

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGET_REGEX,
    task_type="CAUSAL_LM",
)


# %%
# --- Cell 7: BEFORE-training baseline generation (raw model, no LoRA yet) ---
# Captures the base model's behavior on the demo charts; will diff against post-train
# generations from the LoRA-wrapped model in Cell 10.
def vlm_generate(model, processor, demo_row, max_new_tokens=20):
    """Greedy-decode an assistant turn from a {images, prompt, completion} row.

    Uses the row's `prompt` directly (system + user); generation produces the
    assistant turn. Disables grad checkpointing's cache-disable for inference.
    """
    prepared = prepare_multimodal_messages(demo_row["prompt"], images=demo_row["images"])
    prompt = processor.apply_chat_template(prepared, add_generation_prompt=True)
    inputs = processor(
        images=demo_row["images"],
        text=prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    was_training = model.training
    model.eval()
    prev_use_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        model.config.use_cache = prev_use_cache
        if was_training:
            model.train()
    gen_ids = out[0][inputs.input_ids.shape[1] :]
    return processor.batch_decode([gen_ids], skip_special_tokens=True)[0]


before_gens = []
for row in demo_rows:
    gen = vlm_generate(model, processor, row)
    before_gens.append(
        {
            "query": row["prompt"][1]["content"][-1]["text"],
            "expected": row["completion"][0]["content"][0]["text"],
            "generation_base": gen,
        }
    )
(RUN_DIR / "generations_before.json").write_text(json.dumps(before_gens, indent=2))
print(f"wrote {RUN_DIR / 'generations_before.json'}  ({len(before_gens)} demos)")
print(f"  before[0]: q={before_gens[0]['query'][:60]!r}  "
      f"expected={before_gens[0]['expected']!r}  base={before_gens[0]['generation_base']!r}")


# %%
# --- Cell 8: SFTConfig + SFTTrainer (passes raw model + peft_config; TRL wraps) ---
sft_config = SFTConfig(
    output_dir=str(RUN_DIR),
    seed=SEED,
    # CRITICAL: None (not -1) disables truncation so image tokens are not chopped.
    max_length=MAX_LENGTH,
    # Mask everything before the completion (-100). TRL auto-sets this True when the
    # dataset has prompt/completion columns; we set it explicitly for documentation.
    completion_only_loss=True,
    # batch / optimization
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    optim=OPTIM,
    bf16=(USE_BF16 and device == "cuda"),  # CUDA autocast; on MPS dtype drives bf16
    fp16=False,
    gradient_checkpointing=GRAD_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=DATALOADER_WORKERS,
    # duration
    max_steps=SMOKE_MAX_STEPS if SMOKE else -1,
    num_train_epochs=1 if SMOKE else NUM_EPOCHS,
    # logging / eval / saving
    logging_steps=2 if SMOKE else 10,
    eval_strategy="no" if SMOKE else "steps",
    eval_steps=None if SMOKE else 100,
    save_strategy="no" if SMOKE else "steps",
    save_steps=200,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to="none" if SMOKE else REPORT_TO,
    run_name=RUN_NAME,
    remove_unused_columns=False,  # collator needs `images` + `messages`
    disable_tqdm=True,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=processor,  # ProcessorMixin → SFTTrainer flips to VLM mode
    peft_config=peft_config,  # SFTTrainer calls get_peft_model internally
)

# Confirm SFTTrainer detected VLM mode and chose the vision collator.
assert getattr(trainer, "_is_vlm", False), "SFTTrainer did NOT flip to VLM mode"
assert getattr(trainer, "_is_vision_dataset", False), "SFTTrainer did not detect vision dataset"
print(f"  _is_vlm={trainer._is_vlm}  _is_vision_dataset={trainer._is_vision_dataset}")

# Confirm vision tower is frozen (all base params frozen by PEFT after wrapping) and
# no LoRA modules live under vision_model.
vision_trainable = sum(
    p.numel()
    for n, p in trainer.model.named_parameters()
    if "vision_model" in n and p.requires_grad
)
assert vision_trainable == 0, f"vision encoder has {vision_trainable} trainable params (expected 0)"
trainer.model.print_trainable_parameters()

resolved_config = {
    "run_name": RUN_NAME,
    "mode": "smoke" if SMOKE else "scale",
    "seed": SEED,
    "model_id": MODEL_ID,
    "dataset_id": DATASET_ID,
    "device": device,
    "model_dtype": str(model_dtype),
    "do_image_splitting": processor.image_processor.do_image_splitting,
    "lora": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_regex": LORA_TARGET_REGEX,
    },
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
print(f"training: {'SMOKE' if SMOKE else 'SCALE'}  ->  {RUN_DIR}")
t0 = time.time()
train_result = trainer.train()
wall_s = time.time() - t0

log_history = trainer.state.log_history
loss_logs = [e["loss"] for e in log_history if "loss" in e]
assert loss_logs, "no training loss logged — something is wrong with the run"
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
    "first_loss": first_loss,
    "last_loss": last_loss,
    "loss_trend_ok": last_loss < first_loss,
    "loss_history": loss_logs,
    "train_runtime_s": train_result.metrics.get("train_runtime"),
}
(RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"wrote {RUN_DIR / 'metrics.json'}")


# %%
# --- Cell 10: save -> push to Hub -> AFTER-gen (best-effort) ---
# CRITICAL ORDERING (feedback_cloud_training_robustness): adapter ships to the Hub
# FIRST. After-gen is best-effort. On U1's first cloud run we lost a fully-trained
# adapter because the post-train work tipped past the timeout.
trainer.save_model(str(RUN_DIR))
print(f"saved adapter to {RUN_DIR}")

if not SMOKE:
    print(f"\npushing {RUN_DIR} -> {HUB_MODEL_ID} (private={HUB_PRIVATE}) ...")
    create_repo(HUB_MODEL_ID, exist_ok=True, private=HUB_PRIVATE)
    HfApi().upload_folder(
        folder_path=str(RUN_DIR),
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        # Drop checkpoint-*/ dirs (optimizer state, ~hundreds of MB) — U2 lesson.
        ignore_patterns=["checkpoint-*/*", "checkpoint-*"],
        commit_message=f"VLM-SFT LoRA adapter (ChartQA) — run {RUN_NAME}",
    )
    print(f"published: https://huggingface.co/{HUB_MODEL_ID}")

try:
    after_gens = []
    for row in demo_rows:
        gen = vlm_generate(trainer.model, processor, row)
        after_gens.append(
            {
                "query": row["prompt"][1]["content"][-1]["text"],
                "expected": row["completion"][0]["content"][0]["text"],
                "generation_ft": gen,
            }
        )
    (RUN_DIR / "generations_after.json").write_text(json.dumps(after_gens, indent=2))
    print(f"wrote {RUN_DIR / 'generations_after.json'}")

    identical = sum(1 for b, a in zip(before_gens, after_gens) if b["generation_base"] == a["generation_ft"])
    print(f"\n--- before/after diff ({len(after_gens)} demos; {identical} byte-identical) ---")
    for i, (b, a) in enumerate(zip(before_gens, after_gens, strict=True)):
        print(f"  [{i}] q={b['query'][:60]!r}  expected={b['expected']!r}")
        print(f"      base : {b['generation_base']!r}")
        print(f"      ft   : {a['generation_ft']!r}")

    # U2 dead-run guard: a smoke that produces byte-identical before/after = the
    # policy didn't move. Hard-fail the smoke so we don't waste a cloud submission.
    if SMOKE:
        assert identical < len(after_gens), (
            "SMOKE produced byte-identical before/after for ALL demos — policy did not "
            "move. Investigate LR, LoRA scope, or whether trainable params reached the "
            "forward pass before flipping SMOKE=False."
        )

    if not SMOKE:
        HfApi().upload_file(
            path_or_fileobj=str(RUN_DIR / "generations_after.json"),
            path_in_repo="generations_after.json",
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            commit_message=f"add after-training generations for {RUN_NAME}",
        )
        print(f"pushed generations_after.json to {HUB_MODEL_ID}")
except AssertionError:
    raise  # don't swallow the dead-run guard
except Exception as e:
    print(f"\nWARN: after-generation step failed (best-effort): {type(e).__name__}: {e}")
    print("The adapter is already safe locally (and on the Hub if SCALE). Skipping after-gen.")


# %%
# --- Cell 11: runtime gate / run summary ---
if SMOKE:
    scale_train_rows = train_ds_full.num_rows  # full TRAIN_SPLIT for the scale run
    eff_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    scale_steps = ceil(scale_train_rows / eff_batch) * NUM_EPOCHS
    print(f"\nsmoke complete: {sec_per_step:.2f}s/step over {n_steps} steps")
    print(f"SCALE-RUN STEPS: ~{scale_steps:,} (1 epoch over {scale_train_rows:,} rows, eff batch {eff_batch})")
    # Do NOT extrapolate cloud wall-time from MPS smoke — different hardware AND
    # smoke has do_image_splitting=False (much shorter sequences than scale).
    print("  cloud estimate: a10g-large @ ~2-4 s/step -> ~30-90 min, ~$1-3")
    print("\n  before flipping SMOKE=False:")
    print("  - confirm before/after diff is non-trivial (not just byte-different but visibly different)")
    print("  - confirm loss is trending down")
    print(f"  - confirm max_length is None (not -1) in {RUN_DIR / 'config.json'}")
    print("  - top up HF Jobs credit to ~$10-15 (NO_CREDITS-mid-run lesson)")
    print("  - then submit: hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN "
          "--env SMOKE=false <github-raw-url-to-this-file>")
else:
    print(f"\nSCALE run complete: {RUN_NAME}")
    print(f"  wall time: {wall_s / 60:.1f} min  |  loss {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  artifacts in {RUN_DIR}/  (config, metrics, length_stats, generations, adapter)")
    print(f"  published: https://huggingface.co/{HUB_MODEL_ID}")
    print("  next: review trackio dashboard, before/after generations, model card, /review, flip public")

# %%
