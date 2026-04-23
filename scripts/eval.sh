#!/bin/bash

set -e
set -o pipefail

model_base=lmsys/vicuna-7b-v1.5
#model_path=microsoft/llava-rad
model_path=/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/checkpoints/llava-rad/biomedclip_cxr_518-lora-3e-1e-4-20260412014758
#model_path=/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/checkpoints/llava-rad/biomedclip_cxr_518-lora-3e-1e-5-20260417204121

model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/predictions/llava-rad-lora-1e-4}"
prediction_file=$prediction_dir/model_predictions

run_name="${4:-llavarad}"


query_file=/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/data/chexpert/processed/llavarad_valid.jsonl
#query_file=/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/scripts/gpt_processed_data.jsonl

image_folder=/data/raw_data/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0
loader="chexpert_valid_findings_impressions"
conv_mode="v1"

# CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# CHUNKS=1
# for (( idx=0; idx<$CHUNKS; idx++ ))
# do
#     CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_mimic_cxr \
#         --query_file ${query_file} \
#         --loader ${loader} \
#         --image_folder ${image_folder} \
#         --conv_mode ${conv_mode} \
#         --prediction_file ${prediction_file}_${idx}.jsonl \
#         --temperature 0 \
#         --model_path ${model_path} \
#         --model_base ${model_base} \
#         --chunk_idx ${idx} \
#         --num_chunks ${CHUNKS} \
#         --batch_size 8 \
#         --group_by_length &
# done

# wait

cat ${prediction_file}_*.jsonl > chexpert_preds.jsonl

pushd llava/eval/rrg_eval
wandb_run_id="llava-eval-$(date +%Y%m%d%H%M%S)"
WANDB_PROJECT="llava-rad-finetuning" \
WANDB_RUN_ID="${wandb_run_id}" \
WANDB_RUN_GROUP=evaluate \
WANDB_CONSOLE=off \
CUDA_VISIBLE_DEVICES=1 \
python run.py ../../../chexpert_preds.jsonl --run_name ${run_name} --output_dir ${prediction_dir}/eval
popd

rm chexpert_preds.jsonl