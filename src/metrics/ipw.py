"""
ipw.py
------
Metric 3: Intelligence Per Watt (IPW)

IPW = Normalized_Task_Score (0–1) / Energy_per_Request (J)

Normalisation:
  - chat scores are on 1–10 → divide by 10
  - rag / summ scores are already on 0–1

Higher IPW = better task performance per unit of energy consumed.
"""

import pandas as pd

from q_ret import extract_model_components


def normalize_score(score: float, task: str) -> float:
    if task == "chat":
        return score / 10.0
    return score   # rag and summ are already 0–1


def calculate_intelligence_per_watt(
    perf_df:      pd.DataFrame,
    inference_df: pd.DataFrame,
    family_name:  str,
) -> pd.DataFrame:
    """
    Args:
        perf_df:      final_36_variants_report_{family}.csv
        inference_df: vllm_inference_metrics filtered to this family
        family_name:  "Llama" or "Qwen"

    Returns:
        DataFrame with IPW for every model variant.
    """
    print(f"\n{'='*80}")
    print(f"METRIC 3: Intelligence Per Watt (IPW) — {family_name}")
    print(f"{'='*80}")

    perf_df = perf_df.copy()
    perf_df["Components"] = perf_df["Model"].apply(extract_model_components)
    perf_df[["Family_Parsed", "Size", "Method", "Task_Parsed", "Precision_Parsed"]] = pd.DataFrame(
        perf_df["Components"].tolist(), index=perf_df.index
    )

    results = []

    for _, perf_row in perf_df.iterrows():
        size      = perf_row["Size"]
        method    = perf_row["Method"]
        task      = perf_row["Task_Parsed"]
        precision = perf_row["Precision_Parsed"]

        i_mask = (
            (inference_df["Size"] == size) &
            (inference_df["Method"] == method) &
            (inference_df["Task"] == task) &
            (inference_df["Quantization"] == precision)
        )
        i_sub = inference_df[i_mask]
        if len(i_sub) == 0:
            continue

        raw_score        = perf_row["Primary_Average"]
        norm_score       = normalize_score(raw_score, task)
        e_per_req_j      = i_sub["Energy_Total_J"].values[0] / i_sub["Num_Samples"].values[0]
        ipw              = (norm_score / e_per_req_j) if e_per_req_j > 0 else 0.0

        results.append({
            "Family":                family_name,
            "Size":                  size,
            "Method":                method,
            "Task":                  task,
            "Quantization":          precision,
            "Model_Name":            perf_row["Model"],
            "Raw_Score":             raw_score,
            "Normalized_Score_0_1":  norm_score,
            "Energy_per_Req_J":      e_per_req_j,
            "IPW":                   ipw,
            "Power_Avg_W":           i_sub["Power_Avg_W"].values[0],
            "Throughput_tokens_s":   i_sub["Overall_Throughput"].values[0],
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"\n  Total variants analysed: {len(df)}")
        print(f"\n  IPW Statistics:")
        print(f"    Mean:   {df['IPW'].mean():.6f}")
        print(f"    Median: {df['IPW'].median():.6f}")
        print(f"    Min:    {df['IPW'].min():.6f}")
        print(f"    Max:    {df['IPW'].max():.6f}")
        print("\n  By Quantization:")
        for q in ["FP16", "INT8", "INT4"]:
            sub = df[df["Quantization"] == q]
            if len(sub):
                print(f"    {q}: Mean={sub['IPW'].mean():.6f}")
        print("\n  By Size:")
        for s in ["1B", "3B", "7B"]:
            sub = df[df["Size"] == s]
            if len(sub):
                print(f"    {s}: Mean={sub['IPW'].mean():.6f}")

    return df
