"""
rho_sys.py
----------
Metric 4: System Density (ρ_sys)

ρ_sys = Throughput (tokens/s) / Model_Size_in_Memory (GB)

Measures how efficiently a model uses its memory footprint.
Higher is better. A 1B INT4 model typically has 3–4× the density
of a 7B FP16 model, making it preferable for latency-constrained deployments.
"""

import pandas as pd


def calculate_system_density(
    inference_df: pd.DataFrame,
    family_name:  str,
    model_sizes_gb: dict,
) -> pd.DataFrame:
    """
    Args:
        inference_df:   vllm_inference_metrics filtered to this family
        family_name:    "Llama" or "Qwen"
        model_sizes_gb: dict mapping (size_label, quantization) → GB in memory

    Returns:
        DataFrame with ρ_sys for every model variant.
    """
    print(f"\n{'='*80}")
    print(f"METRIC 4: System Density (ρ_sys) — {family_name}")
    print(f"{'='*80}")

    results = []

    for _, row in inference_df.iterrows():
        size  = row["Size"]
        quant = row["Quantization"]
        size_gb = model_sizes_gb.get((size, quant))
        if size_gb is None:
            continue

        throughput = row["Overall_Throughput"]
        rho_sys    = throughput / size_gb if size_gb > 0 else 0.0

        results.append({
            "Family":                 family_name,
            "Size":                   size,
            "Method":                 row["Method"],
            "Task":                   row["Task"],
            "Quantization":           quant,
            "Model_Name":             row["Model_ID"],
            "Throughput_tokens_s":    throughput,
            "Model_Size_GB":          size_gb,
            "rho_sys_tokens_s_per_GB": rho_sys,
            "TTFT_Mean_ms":           row["TTFT_Mean"],
            "ITL_Mean_ms":            row["ITL_Mean"],
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"\n  Total variants analysed: {len(df)}")
        print(f"\n  ρ_sys Statistics:")
        print(f"    Mean:   {df['rho_sys_tokens_s_per_GB'].mean():.0f} tokens/s/GB")
        print(f"    Median: {df['rho_sys_tokens_s_per_GB'].median():.0f} tokens/s/GB")
        print("\n  By Quantization:")
        for q in ["FP16", "INT8", "INT4"]:
            sub = df[df["Quantization"] == q]
            if len(sub):
                print(f"    {q}: Mean={sub['rho_sys_tokens_s_per_GB'].mean():.0f} tok/s/GB")
        print("\n  By Size (INT4 only):")
        int4 = df[df["Quantization"] == "INT4"]
        for s in ["1B", "3B", "7B"]:
            sub = int4[int4["Size"] == s]
            if len(sub):
                print(f"    {s}: Mean={sub['rho_sys_tokens_s_per_GB'].mean():.0f} tok/s/GB")

        # Key comparison: 1B vs 7B density
        i1 = int4[int4["Size"] == "1B"]["rho_sys_tokens_s_per_GB"].mean()
        i7 = int4[int4["Size"] == "7B"]["rho_sys_tokens_s_per_GB"].mean()
        if i7 and i7 > 0:
            print(f"\n  1B is {i1/i7:.1f}× denser than 7B (INT4)")

    return df
