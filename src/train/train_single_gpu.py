"""
train_single_gpu.py
-------------------
Single-GPU LoRA and QLoRA fine-tuning for 1B and 3B models.
Trains all three tasks (rag, summ, chat) sequentially.

Usage:
    HF_TOKEN=<token> python src/train/train_single_gpu.py

Environment variables:
    HF_TOKEN  - HuggingFace API token (required for gated models like LLaMA)
"""

import os
import gc
import csv
import time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # restrict to single GPU
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

from energy_monitor import EnergyMonitor
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
    login(token=hf_token)
else:
    print("❌ No HF_TOKEN found. Set HF_TOKEN environment variable.")

# ─────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────
# (model_id, size_label, do_lora, do_qlora)
EXPERIMENTS = [
    ("Qwen/Qwen2.5-1.5B-Instruct",          "1B", True, True),
    ("Qwen/Qwen2.5-3B-Instruct",             "3B", True, True),
    ("meta-llama/Llama-3.2-1B-Instruct",     "1B", True, True),
    ("meta-llama/Llama-3.2-3B-Instruct",     "3B", True, True),
]

# Training row counts vary by model size to equalise compute
ROW_COUNTS = {
    "1B": {"rag": 15000, "summ": 5000, "chat": 10000},
    "3B": {"rag": 10000, "summ": 5000, "chat": 10000},
}

OUTPUT_DIR = "./fine_tuned_adapters"
LOG_FILE   = "training_energy_log.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# LoRA config (same for both LoRA and QLoRA)
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
            ["Model", "Size", "Method", "Task", "Joules", "Watts", "Seconds", "Rows"]
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
                    model_id, trust_remote_code=True
                )
                tokenizer.pad_token    = tokenizer.eos_token
                tokenizer.padding_side = "right"

                # ── Load model ────────────────────────────────────────────
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
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
                monitor = EnergyMonitor(interval=0.5)
                monitor.start()

                trainer = SFTTrainer(
                    model=model,
                    train_dataset=formatted_dataset,
                    args=TrainingArguments(
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=2,
                        max_steps=-1,
                        num_train_epochs=1,
                        learning_rate=2e-4,
                        fp16=not is_qlora,   # FP16 for LoRA, BF16 for QLoRA
                        bf16=is_qlora,
                        logging_steps=20,
                        output_dir="temp_trainer",
                        optim="paged_adamw_32bit",
                        gradient_checkpointing=True,
                        report_to="none",
                        save_strategy="no",
                    ),
                )

                print("      🚀 Starting training...")
                trainer.train()

                energy = monitor.stop()
                joules   = energy["total_joules"]
                watts    = energy["avg_watts"]
                duration = energy["duration"]
                print(f"      ⚡ Energy: {joules:.2f} J | {watts:.2f} W | {duration:.2f} s")

                # ── Log ───────────────────────────────────────────────────
                with open(LOG_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([
                        model_id, size_label, tune_type, task,
                        f"{joules:.2f}", f"{watts:.2f}", f"{duration:.2f}",
                        len(formatted_dataset),
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
                for obj in ["model", "tokenizer", "trainer", "formatted_dataset"]:
                    if obj in dir():
                        try:
                            del locals()[obj]
                        except Exception:
                            pass
                gc.collect()
                torch.cuda.empty_cache()

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print(f"📊 Results logged to: {LOG_FILE}")
print(f"💾 Adapters saved to: {OUTPUT_DIR}")
