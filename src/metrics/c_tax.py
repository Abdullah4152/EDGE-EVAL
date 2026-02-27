"""
c_tax.py
--------
Metric 5: Cold-Start Tax (C_tax)

C_tax = E_load / E_infer

Where:
  E_load  = energy consumed loading the model (load_time × avg_power_during_load)
  E_infer = energy per single inference request

C_tax > 100 means one cold start costs more energy than 100 inferences.
Such models are labelled "Serverless_Prohibitive" — they should NOT be
deployed in ephemeral / serverless environments (e.g. AWS Lambda, K8s pods
that spin up per request).
"""

import pandas as pd


def calculate_cold_start_tax(
    inference_df: pd.DataFrame,
    family_name:  str,
) -> pd.DataFrame:
    """
    Args:
        inference_df: vllm_inference_metrics filtered to this family
        family_name:  "Llama" or "Qwen"

    Returns:
        DataFrame with C_tax and serverless viability flag for every variant.
    """
    print(f"\n{'='*80}")
    print(f"METRIC 5: Cold-Start Tax (C_tax) — {family_name}")
    print(f"{'='*80}")

    results = []

    for _, row in inference_df.iterrows():
        load_time_s  = row["Model_Load_Time"]
        avg_power_w  = row["Power_Avg_W"]
        e_load_j     = load_time_s * avg_power_w

        total_energy_j = row["Energy_Total_J"]
        num_samples    = row["Num_Samples"]
        e_infer_j      = total_energy_j / num_samples if num_samples > 0 else 0.0

        c_tax = (e_load_j / e_infer_j) if e_infer_j > 0 else 0.0

        results.append({
            "Family":                family_name,
            "Size":                  row["Size"],
            "Method":                row["Method"],
            "Task":                  row["Task"],
            "Quantization":          row["Quantization"],
            "Model_Name":            row["Model_ID"],
            "Load_Time_s":           load_time_s,
            "Avg_Power_W":           avg_power_w,
            "E_load_J":              e_load_j,
            "E_infer_J":             e_infer_j,
            "C_tax":                 c_tax,
            "Serverless_Prohibitive": "Yes" if c_tax > 100 else "No",
            "Equivalent_Inferences": int(c_tax),
        })

    df = pd.DataFrame(results)

    if len(df) > 0:
        print(f"\n  Total variants analysed: {len(df)}")
        print(f"\n  C_tax Statistics:")
        print(f"    Mean:   {df['C_tax'].mean():.0f}×")
        print(f"    Median: {df['C_tax'].median():.0f}×")
        prohibitive = len(df[df["C_tax"] > 100])
        print(f"\n  Serverless Prohibitive (>100×): {prohibitive}/{len(df)}")
        print("\n  By Size:")
        for s in ["1B", "3B", "7B"]:
            sub = df[df["Size"] == s]
            if len(sub):
                print(f"    {s}: Mean={sub['C_tax'].mean():.0f}×  "
                      f"Prohibitive={len(sub[sub['C_tax']>100])}/{len(sub)}")

    return df
