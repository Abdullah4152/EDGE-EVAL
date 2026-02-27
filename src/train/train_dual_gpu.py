"""
train_dual_gpu.py
-----------------
Dual-GPU LoRA and QLoRA fine-tuning for 7B/8B models using device_map="auto".
On 2× Tesla T4 (16 GB each), the model is split automatically across both GPUs.

For 1B and 3B models this script also works but runs all layers on GPU 0.
For those sizes, train_single_gpu.py is preferred.

Usage:
    HF_TOKEN=<token> python src/train/train_dual_gpu.py

Environment variables:
    HF_TOKEN  - HuggingFace API token (required for gated models like LLaMA)
"""

import os
import gc
import csv
import time
import torch

# ⭐ Do NOT set CUDA_VISIBLE_DEVICES — let both GPUs be visible
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

from energy_monitor import DualGPUEnergyMonitor
from data_loader import load_and_prep_data, format_data_using_chat_template

# ─────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    print("✅ Using Kaggle Secrets token")
except Exception:
    hf_token = os.getenv("HF_TOKEN", "")

if hf_token:
    login(token=hf_token, add_to_git_credential=True)
else:
    print("❌ No HF_TOKEN found. Set HF_TOKEN environment variable.")

# ── Verify dual GPU ────────────────────────────────────────────────────────
print(f"🔍 GPUs Available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

# ─────────────────────────────────────────────────────────
# Experiment configuration
# Dual GPU is used only for 7B/8B — those models need both GPUs for QLoRA
# (model_id, size_label, do_lora, do_qlora)
# ─────────────────────────────────────────────────────────
EXPERIMENTS = [
    ("Qwen/Qwen2.5-7B-Instruct",           "7B", False, True),
    ("meta-llama/Llama-3.1-8B-Instruct",   "7B", False, True),
    # Uncomment to include smaller models in the same run:
    # ("Qwen/Qwen2.5-1.5B-Instruct",        "1B", True,  True),
    # ("Qwen/Qwen2.5-3B-Instruct",           "3B", True,  True),
    # ("meta-llama/Llama-3.2-1B-Instruct",  "1B", True,  True),
    # ("meta-llama/Llama-3.2-3B-Instruct",  "3B", True,  True),
]

ROW_COUNTS = {
    "1B": {"rag": 15000, "summ": 5000, "chat": 10000},
    "3B": {"rag": 10000, "summ": 5000, "chat": 10000},
    "7B": {"rag": 5000,  "summ": 5000, "chat": 10000},
}

# Memory allocation per GPU — leave ~3 GB headroom for activations
MAX_MEMORY_PER_GPU = {0: "13GB", 1: "13GB"}

OUTPUT_DIR = "./fine_tuned_adapters"
LOG_FILE   = "training_energy_log.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# LoRA config (shared for LoRA and QLoRA)
# ─────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ─────────────────────────────────────────────────────────
# CSV header
# ─────────────────────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Model", "Size", "Method", "Task",
             "Joules", "Watts", "Seconds", "Rows", "GPUs"]
        )

# ─────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────
for model_id, size_label, do_lora, do_qlora in EXPERIMENTS:

    modes = []
    if do_qlora:
        modes.append("QLoRA_INT4")
    if do_lora:
        modes.append("LoRA_FP16")

    for task in ["rag", "summ", "chat"]:
        n_rows = ROW_COUNTS[size_label][task]

        for tune_type in modes:
            print(f"\n{'='*60}\n🤖 {size_label} | {tune_type} | TASK: {task}\n{'='*60}")

            is_qlora = (tune_type == "QLoRA_INT4")
            bnb_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                if is_qlora
                else None
            )

            monitor = None
            try:
                gc.collect()
                torch.cuda.empty_cache()

                # ── Load tokenizer ────────────────────────────────────────
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    token=hf_token,
                    trust_remote_code=True,
                )
                tokenizer.pad_token    = tokenizer.eos_token
                tokenizer.padding_side = "right"

                # ── Load model with dual-GPU distribution ─────────────────
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",          # ⭐ Auto-distribute across GPUs
                    max_memory=MAX_MEMORY_PER_GPU,
                    token=hf_token,
                    trust_remote_code=True,
                    offload_folder="offload",   # CPU offload safety valve
                )
                print(f"   📊 Model device map: {model.hf_device_map}")

                if is_qlora:
                    model = prepare_model_for_kbit_training(model)

                model = get_peft_model(model, LORA_CONFIG)
                model.print_trainable_parameters()

                # ── Load and format data ──────────────────────────────────
                dataset = load_and_prep_data(task, n_rows, tokenizer)
                if dataset is None or len(dataset) == 0:
                    print("      ⚠️  Skipping due to data loading failure")
                    continue

                print(f"      🔄 Formatting {len(dataset)} examples...")
                formatted_dataset = dataset.map(
                    lambda examples: format_data_using_chat_template(examples, tokenizer, task),
                    batched=True,
                    remove_columns=dataset.column_names,
                    desc="Formatting",
                )
                formatted_dataset = formatted_dataset.filter(
                    lambda x: x["text"] is not None and len(x["text"].strip()) > 0
                )
                print(f"      ✅ {len(formatted_dataset)} valid examples after formatting")

                if len(formatted_dataset) == 0:
                    print("      ⚠️  No valid examples, skipping")
                    continue

                # ── Train ─────────────────────────────────────────────────
                monitor = DualGPUEnergyMonitor(interval=0.5)
                monitor.start()

                trainer = SFTTrainer(
                    model=model,
                    train_dataset=formatted_dataset,
                    args=TrainingArguments(
                        per_device_train_batch_size=1,    # Conservative for 7B across 2×T4
                        gradient_accumulation_steps=8,
                        max_steps=-1,
                        num_train_epochs=1,
                        learning_rate=2e-4,
                        fp16=False,   # QLoRA uses BF16
                        bf16=True,
                        logging_steps=20,
                        output_dir="temp_trainer",
                        optim="paged_adamw_32bit",
                        gradient_checkpointing=True,
                        report_to="none",
                        save_strategy="no",
                        dataloader_num_workers=2,
                        ddp_find_unused_parameters=False,
                    ),
                )

                print(f"      🚀 Starting training on {torch.cuda.device_count()} GPU(s)...")
                trainer.train()

                total_j, avg_w, duration, per_gpu_j, per_gpu_peak = monitor.stop()
                print(f"      ⚡ Total Energy: {total_j:.2f} J | {avg_w:.2f} W | {duration:.2f} s")

                # ── Log ───────────────────────────────────────────────────
                with open(LOG_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([
                        model_id, size_label, tune_type, task,
                        f"{total_j:.2f}", f"{avg_w:.2f}", f"{duration:.2f}",
                        len(formatted_dataset), torch.cuda.device_count(),
                    ])

                # ── Save adapter ──────────────────────────────────────────
                adapter_name = f"{size_label}_{tune_type}_{task}"
                save_path    = os.path.join(OUTPUT_DIR, adapter_name)
                trainer.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"      ✅ Saved: {save_path}")

            except Exception as e:
                import traceback
                print(f"      ❌ FAILED: {e}")
                traceback.print_exc()
                if monitor:
                    try:
                        monitor.stop()
                    except Exception:
                        pass
            finally:
                for name in ["model", "tokenizer", "trainer", "formatted_dataset"]:
                    if name in dir():
                        exec(f"del {name}")
                gc.collect()
                torch.cuda.empty_cache()

print("\n" + "="*60)
print("✅ DUAL-GPU TRAINING COMPLETE!")
print(f"📊 Results logged to: {LOG_FILE}")
print(f"💾 Adapters saved to: {OUTPUT_DIR}")
