"""
q_ret.py
--------
Metric 1: Quantization Fidelity (Q_ret)

Q_ret = (Score_quantized / Score_FP16) × 100%

Measures how much task accuracy is retained after quantization.
A value >= 99.9% indicates effectively lossless compression.
"""

import pandas as pd


def extract_model_components(model_name: str) -> tuple:
    """Parse model name into (family, size, method, task, precision)."""
    parts  = model_name.split("/")[-1]
    family = "Llama" if "Llama" in model_name else "Qwen"

    size = None
    for s in ["1B", "3B", "7B"]:
        if f"-{s}_" in parts or f"_{s}_" in parts:
            size = s
            break

    method = "QLoRA" if "QLoRA" in parts else "LoRA"

    task = None
    for t in ["rag", "summ", "chat"]:
        if f"_{t}-" in parts or f"_{t}" in parts.lower():
            task = t
            break

    if parts.endswith("-MERGED"):
        precision = "INT4"
    elif parts.endswith("-INT8"):
        precision = "INT8"
    elif parts.endswith("-INT4"):
        precision = "INT4"
    elif parts.endswith("-FP16"):
        precision = "FP16"
    else:
        precision = "Unknown"

    return family, size, method, task, precision


def calculate_quantization_fidelity(perf_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    """
    Compute Q_ret for all quantized variants relative to their FP16 baseline.

    Args:
        perf_df:     DataFrame from final_36_variants_report_{family}.csv
        family_name: "Llama" or "Qwen"

    Returns:
        DataFrame with one row per (size, method, task, quantization) combination.
    """
    print(f"\n{'='*80}")
    print(f"METRIC 1: Quantization Fidelity (Q_ret) — {family_name}")
    print(f"{'='*80}")

    # Parse model name components
    perf_df["Components"] = perf_df["Model"].apply(extract_model_components)
    perf_df[["Family_Parsed", "Size", "Method", "Task_Parsed", "Precision_Parsed"]] = pd.DataFrame(
        perf_df["Components"].tolist(), index=perf_df.index
    )

    results = []

    for size in ["1B", "3B", "7B"]:
        for method in ["LoRA", "QLoRA"]:
            for task in ["rag", "summ", "chat"]:

                # Get FP16 baseline
                fp16_mask = (
                    (perf_df["Size"] == size) &
                    (perf_df["Method"] == method) &
                    (perf_df["Task_Parsed"] == task) &
                    (perf_df["Precision_Parsed"] == "FP16")
                )
                fp16_subset = perf_df[fp16_mask]
                if len(fp16_subset) == 0:
                    continue

                fp16_score = fp16_subset["Primary_Average"].values[0]
                fp16_model = fp16_subset["Model"].values[0]

                # Compare all quantized versions
                quant_mask = (
                    (perf_df["Size"] == size) &
                    (perf_df["Method"] == method) &
                    (perf_df["Task_Parsed"] == task) &
                    (perf_df["Precision_Parsed"].isin(["INT4", "INT8"]))
                )
                for _, row in perf_df[quant_mask].iterrows():
                    quant_score = row["Primary_Average"]
                    q_ret       = (quant_score / fp16_score * 100) if fp16_score > 0 else 0.0

                    results.append({
                        "Family":       family_name,
                        "Size":         size,
                        "Method":       method,
                        "Task":         task,
                        "Quantization": row["Precision_Parsed"],
                        "Model_Name":   row["Model"],
                        "FP16_Baseline":fp16_model,
                        "FP16_Score":   fp16_score,
                        "Quantized_Score": quant_score,
                        "Q_ret_%":      q_ret,
                        "Score_Diff":   fp16_score - quant_score,
                        "Passes_99.9%": "Yes" if q_ret >= 99.9 else "No",
                    })

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"\n  Total variants analysed: {len(df)}")
        print("\n  By Quantization:")
        for q in df["Quantization"].unique():
            sub = df[df["Quantization"] == q]
            print(f"    {q}: Mean={sub['Q_ret_%'].mean():.2f}%  "
                  f"Min={sub['Q_ret_%'].min():.2f}%  Max={sub['Q_ret_%'].max():.2f}%")
        print("\n  By Size:")
        for s in ["1B", "3B", "7B"]:
            sub = df[df["Size"] == s]
            if len(sub):
                print(f"    {s}: Mean={sub['Q_ret_%'].mean():.2f}%")

    return df
