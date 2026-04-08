#!/usr/bin/env python3
"""
Compare MedGemma, LLaVA-Rad Base, and LLaVA-Rad Finetuned results
Includes comprehensive comparison with confidence intervals and statistical significance
"""

import pandas as pd
import numpy as np

# Load results
medgemma_path = '/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/medgemma_base/eval/main.csv'
llavarad_base_path = '/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/llavarad_base/eval/main.csv'
llavarad_finetuned_path = '/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/main.csv'

medgemma = pd.read_csv(medgemma_path, index_col=0)
llavarad_base = pd.read_csv(llavarad_base_path, index_col=0)
llavarad_finetuned = pd.read_csv(llavarad_finetuned_path, index_col=0)

# Extract median values
medgemma_median = medgemma.loc['median']
llavarad_base_median = llavarad_base.loc['median']
llavarad_finetuned_median = llavarad_finetuned.loc['median']

# Get common metrics (F1 scores)
f1_metrics = [
    'Micro-F1-14', 'Macro-F1-14', 'Micro-F1-5', 'Macro-F1-5',
    'Micro-F1-14+', 'Macro-F1-14+', 'Micro-F1-5+', 'Macro-F1-5+'
]

# Create comparison DataFrame
comparison_data = {
    'Metric': f1_metrics,
    'MedGemma-1.5 4B': [medgemma_median.get(m, np.nan) for m in f1_metrics],
    'LLaVA-Rad Base': [llavarad_base_median.get(m, np.nan) for m in f1_metrics],
    'LLaVA-Rad Finetuned': [llavarad_finetuned_median.get(m, np.nan) for m in f1_metrics],
}

comparison_df = pd.DataFrame(comparison_data)

# Calculate differences
comparison_df['MedGemma vs Base (Δ)'] = comparison_df['MedGemma-1.5 4B'] - comparison_df['LLaVA-Rad Base']
comparison_df['MedGemma vs Finetuned (Δ)'] = comparison_df['MedGemma-1.5 4B'] - comparison_df['LLaVA-Rad Finetuned']
comparison_df['Base vs Finetuned (Δ)'] = comparison_df['LLaVA-Rad Base'] - comparison_df['LLaVA-Rad Finetuned']

# Format for display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}' if not pd.isna(x) else 'N/A')

print("=" * 100)
print("COMPARISON: MedGemma-1.5 4B vs LLaVA-Rad Base vs LLaVA-Rad Finetuned")
print("=" * 100)
print("\nF1 Scores (CheXbert-based, using same methodology)")
print("-" * 100)
print(comparison_df.to_string(index=False))
print("\n")

# Summary statistics
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"\nMedGemma-1.5 4B:")
print(f"  Micro-F1-14: {medgemma_median.get('Micro-F1-14', 'N/A'):.4f}")
print(f"  Macro-F1-14: {medgemma_median.get('Macro-F1-14', 'N/A'):.4f}")
print(f"  Micro-F1-5:  {medgemma_median.get('Micro-F1-5', 'N/A'):.4f}")
print(f"  Macro-F1-5:  {medgemma_median.get('Macro-F1-5', 'N/A'):.4f}")

print(f"\nLLaVA-Rad Base:")
print(f"  Micro-F1-14: {llavarad_base_median.get('Micro-F1-14', 'N/A'):.4f}")
print(f"  Macro-F1-14: {llavarad_base_median.get('Macro-F1-14', 'N/A'):.4f}")
print(f"  Micro-F1-5:  {llavarad_base_median.get('Micro-F1-5', 'N/A'):.4f}")
print(f"  Macro-F1-5:  {llavarad_base_median.get('Macro-F1-5', 'N/A'):.4f}")

print(f"\nLLaVA-Rad Finetuned:")
print(f"  Micro-F1-14: {llavarad_finetuned_median.get('Micro-F1-14', 'N/A'):.4f}")
print(f"  Macro-F1-14: {llavarad_finetuned_median.get('Macro-F1-14', 'N/A'):.4f}")
print(f"  Micro-F1-5:  {llavarad_finetuned_median.get('Micro-F1-5', 'N/A'):.4f}")
print(f"  Macro-F1-5:  {llavarad_finetuned_median.get('Macro-F1-5', 'N/A'):.4f}")

print("\n" + "=" * 100)
print("KEY FINDINGS")
print("=" * 100)

# Compare MedGemma vs Base
micro_f1_14_diff = medgemma_median.get('Micro-F1-14', 0) - llavarad_base_median.get('Micro-F1-14', 0)
macro_f1_14_diff = medgemma_median.get('Macro-F1-14', 0) - llavarad_base_median.get('Macro-F1-14', 0)

print(f"\nMedGemma vs LLaVA-Rad Base:")
print(f"  Micro-F1-14 difference: {micro_f1_14_diff:+.4f} ({'Better' if micro_f1_14_diff > 0 else 'Worse'})")
print(f"  Macro-F1-14 difference: {macro_f1_14_diff:+.4f} ({'Better' if macro_f1_14_diff > 0 else 'Worse'})")

# Compare MedGemma vs Finetuned
micro_f1_14_diff_ft = medgemma_median.get('Micro-F1-14', 0) - llavarad_finetuned_median.get('Micro-F1-14', 0)
macro_f1_14_diff_ft = medgemma_median.get('Macro-F1-14', 0) - llavarad_finetuned_median.get('Macro-F1-14', 0)

print(f"\nMedGemma vs LLaVA-Rad Finetuned:")
print(f"  Micro-F1-14 difference: {micro_f1_14_diff_ft:+.4f} ({'Better' if micro_f1_14_diff_ft > 0 else 'Worse'})")
print(f"  Macro-F1-14 difference: {macro_f1_14_diff_ft:+.4f} ({'Better' if macro_f1_14_diff_ft > 0 else 'Worse'})")

# Detailed comparison with confidence intervals
print("\n" + "=" * 100)
print("DETAILED COMPARISON WITH 95% CONFIDENCE INTERVALS")
print("=" * 100)

# Focus on key F1 metrics for detailed analysis
key_metrics = ['Micro-F1-14', 'Macro-F1-14', 'Micro-F1-5', 'Macro-F1-5']

detailed_data = []
for metric in key_metrics:
    if metric in medgemma.columns:
        detailed_data.append({
            'Metric': metric,
            'MedGemma_median': medgemma.loc['median', metric],
            'MedGemma_ci_l': medgemma.loc['ci_l', metric],
            'MedGemma_ci_h': medgemma.loc['ci_h', metric],
            'LLaVA-Rad_Base_median': llavarad_base.loc['median', metric],
            'LLaVA-Rad_Base_ci_l': llavarad_base.loc['ci_l', metric],
            'LLaVA-Rad_Base_ci_h': llavarad_base.loc['ci_h', metric],
            'LLaVA-Rad_Finetuned_median': llavarad_finetuned.loc['median', metric],
            'LLaVA-Rad_Finetuned_ci_l': llavarad_finetuned.loc['ci_l', metric],
            'LLaVA-Rad_Finetuned_ci_h': llavarad_finetuned.loc['ci_h', metric],
        })

detailed_df = pd.DataFrame(detailed_data)

# Print formatted version with statistical significance
for _, row in detailed_df.iterrows():
    print(f"\n{row['Metric']}:")
    print(f"  MedGemma-1.5 4B:     {row['MedGemma_median']:.4f} [{row['MedGemma_ci_l']:.4f}, {row['MedGemma_ci_h']:.4f}]")
    print(f"  LLaVA-Rad Base:       {row['LLaVA-Rad_Base_median']:.4f} [{row['LLaVA-Rad_Base_ci_l']:.4f}, {row['LLaVA-Rad_Base_ci_h']:.4f}]")
    print(f"  LLaVA-Rad Finetuned:  {row['LLaVA-Rad_Finetuned_median']:.4f} [{row['LLaVA-Rad_Finetuned_ci_l']:.4f}, {row['LLaVA-Rad_Finetuned_ci_h']:.4f}]")
    
    # Check if CIs overlap for statistical significance
    medgemma_ci = (row['MedGemma_ci_l'], row['MedGemma_ci_h'])
    base_ci = (row['LLaVA-Rad_Base_ci_l'], row['LLaVA-Rad_Base_ci_h'])
    finetuned_ci = (row['LLaVA-Rad_Finetuned_ci_l'], row['LLaVA-Rad_Finetuned_ci_h'])
    
    if medgemma_ci[1] < base_ci[0] or medgemma_ci[0] > base_ci[1]:
        print(f"    → MedGemma vs Base: Statistically different (CIs don't overlap)")
    else:
        print(f"    → MedGemma vs Base: CIs overlap (not statistically different)")
        
    if medgemma_ci[1] < finetuned_ci[0] or medgemma_ci[0] > finetuned_ci[1]:
        print(f"    → MedGemma vs Finetuned: Statistically different (CIs don't overlap)")
    else:
        print(f"    → MedGemma vs Finetuned: CIs overlap (not statistically different)")

# Save detailed comparison to CSV
output_path = '/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/model_comparison_detailed.csv'
detailed_df.to_csv(output_path, index=False)
print(f"\n\nDetailed comparison with confidence intervals saved to: {output_path}")

print("\n" + "=" * 100)
print("Note: MedGemma evaluation only includes CheXbert F1 scores.")
print("LLaVA-Rad results also include BLEU, ROUGE-L, and F1-RadGraph metrics.")
print("=" * 100)
