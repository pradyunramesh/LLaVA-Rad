#!/usr/bin/env python3
"""
Run MedGemma evaluation using the EXACT same methodology as LLaVA-Rad.
This uses CheXbert to label BOTH predictions and references, then compares them.
"""

import json
import os
import sys

# Set Hugging Face cache to a location with more disk space
os.environ['HF_HOME'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface'

# Use GPU 5 which is free (CheXbert will use cuda:0 which will map to GPU 5)
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Add the llava eval path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llava', 'eval', 'rrg_eval'))

from run import main as run_evaluation

def run_medgemma_llavarad_evaluation():
    """
    Run MedGemma evaluation using LLaVA-Rad's official evaluation pipeline.
    """
    # Path to MedGemma predictions in JSONL format
    predictions_file = '/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/medgemma_base/predictions.jsonl'
    
    # Output directory for results
    output_dir = '/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/medgemma_base/eval'
    
    # Run name
    run_name = 'medgemma-base'
    
    print("="*80)
    print("Running MedGemma evaluation using LLaVA-Rad's official methodology")
    print("="*80)
    print(f"Predictions file: {predictions_file}")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {run_name}")
    print("="*80)
    
    # Run the evaluation
    # Using only CheXbert for now (F1-RadGraph has download issues)
    # This gives us the critical F1 scores that match LLaVA-Rad methodology
    run_evaluation(
        filepath=predictions_file,
        scorers=['CheXbert'],  # Focus on CheXbert F1 scores which are the main comparison metric
        report_chexbert_f1=True,
        bootstrap_ci=True,
        output_dir=output_dir,
        run_name=run_name
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}/main.csv")
    print("="*80)

if __name__ == "__main__":
    run_medgemma_llavarad_evaluation()
