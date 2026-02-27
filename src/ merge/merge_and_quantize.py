"""
merge_and_quantize.py
---------------------
Merge LoRA/QLoRA adapters into base models and produce quantized variants.

For each LoRA_FP16 adapter → produces 3 variants: FP16, INT8, INT4
For each QLoRA_INT4 adapter → produces 1 variant: MERGED (FP16 after merge)

All merged models are uploaded directly to HuggingFace Hub to avoid
local disk constraints ("zero-disk strategy").

Energy consumption for every merge/quantize operation is logged to
merging_quantization_energy.csv.

Usage:
    python src/merge/merge_and_quantize.py

Required environment variables (or Kaggle Secrets):
    QWEN_HF_TOKEN, QWEN_HF_USERNAME
    LLAMA_HF_TOKEN, LLAMA_HF_USERNAME
"""

import os
import gc
import csv
import shutil
import tempfile
import torch
import pynvml

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login, create_repo, upload_folder, HfApi

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from energy_monitor import DualGPUEnergyMonitor

# ─────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────
QWEN_HF_TOKEN    = ""
QWEN_HF_USERNAME = ""
LLAMA_HF_TOKEN   = ""
LLAMA_HF_USERNAME= ""

try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    QWEN_HF_TOKEN     = _s.get_secret("QWEN_HF_TOKEN")
    QWEN_HF_USERNAME  = _s.get_secret("QWEN_HF_USERNAME")
    LLAMA_HF_TOKEN    = _s.get_secret("LLAMA_HF_TOKEN")
    LLAMA_HF_USERNAME = _s.get_secret("LLAMA_HF_USERNAME")
    print("✅ Credentials loaded from Kaggle Secrets")
except Exception:
    QWEN_HF_TOKEN     = os.getenv("QWEN_HF_TOKEN",     "")
    QWEN_HF_USERNAME  = os.getenv("QWEN_HF_USERNAME",  "")
    LLAMA_HF_TOKEN    = os.getenv("LLAMA_HF_TOKEN",    "")
    LLAMA_HF_USERNAME = os.getenv("LLAMA_HF_USERNAME", "")
    print("⚠️  Using environment variables for credentials")

if QWEN_HF_TOKEN:
    login(token=QWEN_HF_TOKEN)

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
QWEN_ADAPTERS_DIR  = os.getenv("QWEN_ADAPTERS_DIR",  "./fine_tuned_adapters/qwen")
LLAMA_ADAPTERS_DIR = os.getenv("LLAMA_ADAPTERS_DIR", "./fine_tuned_adapters/llama")
QWEN_CSV           = os.getenv("QWEN_CSV",  "./training_energy_log_qwen.csv")
LLAMA_CSV          = os.getenv("LLAMA_CSV", "./training_energy_log_llama.csv")
ENERGY_LOG         = "./merging_quantization_energy.csv"
TEMP_DIR           = "/tmp/model_temp"

# ─────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────
def detect_gpus() -> int:
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        print(f"\n🎮 Detected {n} GPU(s):")
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            print(f"   GPU {i}: {pynvml.nvmlDeviceGetName(h)}")
        return n
    except Exception:
        return 0

NUM_GPUS = detect_gpus()

# ─────────────────────────────────────────────────────────
# Disk / memory utilities
# ─────────────────────────────────────────────────────────
def get_dir_size_gb(path: str) -> float:
    total = 0
    if not os.path.exists(path):
        return 0.0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)

def cleanup_directory(path: str, name: str = "") -> float:
    if os.path.exists(path):
        try:
            freed = get_dir_size_gb(path)
            shutil.rmtree(path)
            print(f"      ✓ Cleaned {name}: {freed:.2f} GB freed")
            return freed
        except Exception as e:
            print(f"      ✗ Failed to clean {name}: {e}")
    return 0.0

def aggressive_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()

def full_cleanup():
    aggressive_memory_cleanup()
    cleanup_directory(TEMP_DIR, "/tmp/model_temp")
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    for sub in ["hub", "transformers", "modules"]:
        cleanup_directory(os.path.join(hf_cache, sub), f"HF {sub} cache")
    cleanup_directory(os.path.expanduser("~/.cache/torch"), "Torch cache")
    aggressive_memory_cleanup()

# ─────────────────────────────────────────────────────────
# HuggingFace helpers
# ─────────────────────────────────────────────────────────
def get_already_uploaded_models(username: str, token: str) -> set:
    uploaded = set()
    try:
        api = HfApi()
        for repo in api.list_models(author=username, token=token):
            uploaded.add(repo.id.split("/")[-1])
        print(f"   ✅ Found {len(uploaded)} already-uploaded models for {username}")
    except Exception as e:
        print(f"   ⚠️  Could not fetch uploaded models: {e}")
    return uploaded

def get_base_model_id(adapter_name: str, csv_path: str) -> str | None:
    try:
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                if f"{row['Size']}_{row['Method']}_{row['Task']}" == adapter_name:
                    return row["Model"]
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────
# Core merge + upload
# ─────────────────────────────────────────────────────────
def merge_and_upload(adapter_path: str, base_model_id: str, repo_id: str,
                     hf_token: str, is_qlora: bool = False):
    """Merge adapter weights into base model (FP16) and upload."""
    kind = "QLoRA" if is_qlora else "LoRA"
    print(f"   🔗 Merging {kind} adapter → {repo_id}")

    monitor = DualGPUEnergyMonitor()
    monitor.start()
    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        base  = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, token=hf_token,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        model.save_pretrained(TEMP_DIR, safe_serialization=True)
        tokenizer.save_pretrained(TEMP_DIR)
        del base, model, tokenizer
        aggressive_memory_cleanup()

        try:
            create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
        except Exception:
            pass
        upload_folder(folder_path=TEMP_DIR, repo_id=repo_id, token=hf_token,
                      commit_message="Upload merged model")
        print("   ✅ Uploaded!")
        result = monitor.stop()
        full_cleanup()
        return (True, *result)
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        monitor.stop()
        full_cleanup()
        return (False, 0, 0, 0, [], [])


def quantize_and_upload(adapter_path: str, base_model_id: str, repo_id: str,
                        hf_token: str, quant_type: str = "int8"):
    """Merge LoRA, apply BitsAndBytes quantization (INT8 or INT4), and upload."""
    print(f"   🔨 Merge + Quantize → {quant_type.upper()} → {repo_id}")

    monitor = DualGPUEnergyMonitor()
    monitor.start()
    temp_fp16  = os.path.join(TEMP_DIR, "fp16")
    temp_quant = os.path.join(TEMP_DIR, "quant")
    os.makedirs(temp_fp16, exist_ok=True)
    os.makedirs(temp_quant, exist_ok=True)

    try:
        # Step 1: merge to FP16
        base  = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, token=hf_token,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model.save_pretrained(temp_fp16, safe_serialization=True)
        tokenizer.save_pretrained(temp_fp16)
        del base, model, tokenizer
        aggressive_memory_cleanup()

        # Step 2: quantize from FP16
        if quant_type == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:  # int4
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            temp_fp16, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(temp_fp16)
        model.save_pretrained(temp_quant, safe_serialization=True)
        tokenizer.save_pretrained(temp_quant)
        del model, tokenizer
        aggressive_memory_cleanup()
        shutil.rmtree(temp_fp16, ignore_errors=True)

        try:
            create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
        except Exception:
            pass
        upload_folder(folder_path=temp_quant, repo_id=repo_id, token=hf_token,
                      commit_message=f"Upload {quant_type.upper()} quantized model")
        print("   ✅ Uploaded!")
        result = monitor.stop()
        full_cleanup()
        return (True, *result)
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        monitor.stop()
        full_cleanup()
        return (False, 0, 0, 0, [], [])

# ─────────────────────────────────────────────────────────
# CSV initialisation
# ─────────────────────────────────────────────────────────
if not os.path.exists(ENERGY_LOG):
    with open(ENERGY_LOG, "w", newline="") as f:
        header = ["Adapter_Name", "Base_Model", "Adapter_Type", "Operation",
                  "Precision", "Total_Joules", "Avg_Watts", "Seconds",
                  "HF_Repo", "Status"]
        for i in range(max(NUM_GPUS, 2)):
            header += [f"GPU{i}_Joules", f"GPU{i}_Peak_Watts"]
        csv.writer(f).writerow(header)

# ─────────────────────────────────────────────────────────
# Per-model dispatcher
# ─────────────────────────────────────────────────────────
def process_single_model(adapter_name, adapter_path, base_model_id,
                          model_family, hf_token, hf_username,
                          uploaded_models, variant_type):
    repo_name = f"{model_family}-{adapter_name}-{variant_type}"
    repo_id   = f"{hf_username}/{repo_name}"

    if repo_name in uploaded_models:
        print(f"   ⏭️  SKIP {variant_type} — already uploaded")
        return True

    print(f"\n   ┌{'─'*53}┐\n   │ 🚀 {variant_type:<48} │")
    print(f"   └{'─'*53}┘")

    if variant_type == "FP16":
        success, j, w, t, gpu_j, gpu_peak = merge_and_upload(
            adapter_path, base_model_id, repo_id, hf_token, is_qlora=False)
    elif variant_type in ("INT8", "INT4"):
        success, j, w, t, gpu_j, gpu_peak = quantize_and_upload(
            adapter_path, base_model_id, repo_id, hf_token, variant_type.lower())
    elif variant_type == "MERGED":
        success, j, w, t, gpu_j, gpu_peak = merge_and_upload(
            adapter_path, base_model_id, repo_id, hf_token, is_qlora=True)
    else:
        return False

    if success:
        adapter_type = "QLoRA_INT4" if variant_type == "MERGED" else "LoRA_FP16"
        operation    = "MERGE" if variant_type in ("FP16", "MERGED") else "QUANTIZE"
        row = [adapter_name, base_model_id, adapter_type, operation, variant_type,
               f"{j:.2f}", f"{w:.2f}", f"{t:.2f}", repo_name, "SUCCESS"]
        for i in range(max(NUM_GPUS, 2)):
            row += ([f"{gpu_j[i]:.2f}", f"{gpu_peak[i]:.2f}"]
                    if i < len(gpu_j) else ["0.00", "0.00"])
        with open(ENERGY_LOG, "a", newline="") as f:
            csv.writer(f).writerow(row)
    return success


def process_adapters(adapters_dir, csv_path, model_family, hf_token, hf_username):
    print(f"\n🔍 Checking uploaded models for {hf_username}...")
    uploaded = get_already_uploaded_models(hf_username, hf_token)

    adapter_folders = [d for d in os.listdir(adapters_dir)
                       if os.path.isdir(os.path.join(adapters_dir, d))]
    lora_adapters  = sorted(a for a in adapter_folders if "LoRA_FP16"  in a)
    qlora_adapters = sorted(a for a in adapter_folders if "QLoRA_INT4" in a)

    for idx, name in enumerate(lora_adapters, 1):
        print(f"\n{'='*60}\n📦 LoRA {idx}/{len(lora_adapters)}: {name}\n{'='*60}")
        path   = os.path.join(adapters_dir, name)
        base   = get_base_model_id(name, csv_path)
        if base:
            for v in ("FP16", "INT8", "INT4"):
                process_single_model(name, path, base, model_family, hf_token, hf_username, uploaded, v)

    for idx, name in enumerate(qlora_adapters, 1):
        print(f"\n{'='*60}\n📦 QLoRA {idx}/{len(qlora_adapters)}: {name}\n{'='*60}")
        path = os.path.join(adapters_dir, name)
        base = get_base_model_id(name, csv_path)
        if base:
            process_single_model(name, path, base, model_family, hf_token, hf_username, uploaded, "MERGED")

# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60 + "\n🚀 MERGE & QUANTIZE — Zero-Disk HF Upload Strategy\n" + "="*60)

    print("\n" + "="*60 + "\nProcessing Qwen models\n" + "="*60)
    process_adapters(QWEN_ADAPTERS_DIR, QWEN_CSV, "Qwen", QWEN_HF_TOKEN, QWEN_HF_USERNAME)

    print("\n" + "="*60 + "\nProcessing Llama models\n" + "="*60)
    process_adapters(LLAMA_ADAPTERS_DIR, LLAMA_CSV, "Llama", LLAMA_HF_TOKEN, LLAMA_HF_USERNAME)

    print("\n" + "="*60 + "\n✅ COMPLETE!\n" + "="*60)
