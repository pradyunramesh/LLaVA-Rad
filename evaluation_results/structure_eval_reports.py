import pandas as pd
import json

# Define file paths for the base location
input_file = '/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/predictions.jsonl'
generated_reports_file = '/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/generated_reports.csv'
ground_truth_file = '/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/ground_truth.csv'

#Define file paths for the fine-tuned location
# input_file = '/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/predictions.jsonl'
# generated_reports_file = '/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/generated_reports.csv'
# ground_truth_file = '/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/ground_truth.csv'

generated_records = []
ground_truth_records = []
with open(input_file, 'r') as f:
    for i, line in enumerate(f,1):
        d = json.loads(line)
        generated_records.append({
            'study_id': i,
            "path_to_image": d['image'],
            "report": d['prediction'], 
            "loss": d['generation loss']
        })
        ground_truth_records.append({
            'study_id': i,
            "path_to_image": d['image'],
            "report": d['reference']
        })

generated_df = pd.DataFrame(generated_records)
ground_truth_df = pd.DataFrame(ground_truth_records)
generated_df.to_csv(generated_reports_file, index=False)
ground_truth_df.to_csv(ground_truth_file, index=False)
print(f"Generated reports saved to {generated_reports_file}")
print(f"Ground truth reports saved to {ground_truth_file}")
    
