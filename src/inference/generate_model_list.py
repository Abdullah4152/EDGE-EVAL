"""
generate_model_list.py
----------------------
Generate the full configuration list for all 72 model variants:
  - 2 families (Llama, Qwen)
  - 3 sizes each (1B, 3B, 7B)
  - 3 tasks each (rag, summ, chat)
  - 4 quantization variants each (FP16, INT8, INT4, MERGED)

  2 × 3 × 3 × 4 = 72 variants

Also patches the Qwen2 tokenizer to handle malformed extra_special_tokens.
"""

import os
from typing import List, Dict

# ─────────────────────────────────────────────────────────
# HuggingFace tokens (loaded from environment)
# ─────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    QWEN_HF_TOKEN  = _s.get_secret("QWEN_HF_TOKEN")
    LLAMA_HF_TOKEN = _s.get_secret("LLAMA_HF_TOKEN")
    HF_USERNAME    = _s.get_secret("HF_USERNAME")
except Exception:
    QWEN_HF_TOKEN  = os.getenv("QWEN_HF_TOKEN",  "")
    LLAMA_HF_TOKEN = os.getenv("LLAMA_HF_TOKEN", "")
    HF_USERNAME    = os.getenv("HF_USERNAME",     "YOUR_HF_USERNAME")

# ─────────────────────────────────────────────────────────
# Qwen tokenizer patch (handles list instead of dict in extra_special_tokens)
# ─────────────────────────────────────────────────────────
def patch_qwen2_tokenizer():
    try:
        from transformers.models.qwen2 import tokenization_qwen2_fast
        original_init = tokenization_qwen2_fast.Qwen2TokenizerFast.__init__

        def patched_init(self, *args, **kwargs):
            if isinstance(kwargs.get("extra_special_tokens"), list):
                kwargs["extra_special_tokens"] = {}
            original_init(self, *args, **kwargs)

        tokenization_qwen2_fast.Qwen2TokenizerFast.__init__ = patched_init
        print("✅ Qwen2 tokenizer patch applied")
    except Exception as e:
        print(f"⚠️  Could not patch Qwen2 tokenizer: {e}")


patch_qwen2_tokenizer()

# ─────────────────────────────────────────────────────────
# Model list generator
# ─────────────────────────────────────────────────────────
def generate_all_models(hf_username: str = HF_USERNAME) -> List[Dict]:
    """
    Generate the list of all 72 model variant configs.
    Each entry contains everything needed to run vLLM inference.

    Note: INT8 and INT4 quantization is auto-detected by vLLM from
    the model config — quantization=None means vLLM reads the saved config.
    QLoRA MERGED models are stored as FP16 on disk (merge_and_unload was called).
    """
    models = []

    # ── Qwen: 3 sizes × 3 tasks × 4 variants = 36 ─────────────────────────
    for size in ["1B", "3B", "7B"]:
        for task in ["rag", "summ", "chat"]:
            base = f"{hf_username}/Qwen-{size}"

            models.append({
                "model_id":    f"{base}_LoRA_FP16_{task}-FP16",
                "family":      "qwen", "size": size, "method": "LoRA",
                "precision":   "FP16", "task": task,
                "quantization": None,   # no vLLM quantization flag
                "token":       QWEN_HF_TOKEN,
            })
            models.append({
                "model_id":    f"{base}_LoRA_FP16_{task}-INT8",
                "family":      "qwen", "size": size, "method": "LoRA",
                "precision":   "INT8", "task": task,
                "quantization": None,   # vLLM auto-detects bitsandbytes config
                "token":       QWEN_HF_TOKEN,
            })
            models.append({
                "model_id":    f"{base}_LoRA_FP16_{task}-INT4",
                "family":      "qwen", "size": size, "method": "LoRA",
                "precision":   "INT4", "task": task,
                "quantization": None,   # vLLM auto-detects bitsandbytes config
                "token":       QWEN_HF_TOKEN,
            })
            models.append({
                "model_id":    f"{base}_QLoRA_INT4_{task}-MERGED",
                "family":      "qwen", "size": size, "method": "QLoRA",
                "precision":   "INT4", "task": task,
                "quantization": None,   # FP16 model (merged), no extra quant
                "token":       QWEN_HF_TOKEN,
            })

    # ── Llama: 3 sizes × 3 tasks × 4 variants = 36 ────────────────────────
    for size in ["1B", "3B", "7B"]:
        for task in ["rag", "summ", "chat"]:
            base = f"{hf_username}/Llama-{size}"

            models.append({
                "model_id":    f"{base}_LoRA_FP16_{task}-FP16",
                "family":      "llama", "size": size, "method": "LoRA",
                "precision":   "FP16", "task": task,
                "quantization": None,
                "token":       LLAMA_HF_TOKEN,
            })
            models.append({
                "model_id":    f"{base}_LoRA_FP16_{task}-INT8",
                "family":      "llama", "size": size, "method": "LoRA",
                "precision":   "INT8", "task": task,
                "quantization": None,
                "token":       LLAMA_HF_TOKEN,
            })
            models.append({
                "model_id":    f"{base}_LoRA_FP16_{task}-INT4",
                "family":      "llama", "size": size, "method": "LoRA",
                "precision":   "INT4", "task": task,
                "quantization": None,
                "token":       LLAMA_HF_TOKEN,
            })
            models.append({
                "model_id":    f"{base}_QLoRA_INT4_{task}-MERGED",
                "family":      "llama", "size": size, "method": "QLoRA",
                "precision":   "INT4", "task": task,
                "quantization": None,
                "token":       LLAMA_HF_TOKEN,
            })

    print(f"✅ Generated {len(models)} model configs")
    return models


if __name__ == "__main__":
    models = generate_all_models()
    print(f"\nExample entry:\n  {models[0]}")
    print(f"\nTotal: {len(models)} variants")
