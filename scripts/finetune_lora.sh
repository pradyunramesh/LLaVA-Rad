#!/bin/bash

set -euo pipefail

# Always run from repository root so relative paths resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer this repository's llava package over any site-packages installation.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Ensure CUDA runtime libraries from the active conda env are discoverable.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
model_base=lmsys/vicuna-7b-v1.5
output_dir="/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/checkpoints/llava-rad"

#PROJECTOR="/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/checkpoints/biomedclip_cxr_518-pt-1e-1e-3-20250907160427/mm_projector.bin" # generated using pretrain.sh - Keep empty to make use of the pretrained projector
PROJECTOR="/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/checkpoints/mm_projector.bin" # generated using pretrain.sh - Keep empty to make use of the pretrained projector
vision_tower="biomedclip_cxr_518"
vision_tower_config="biomedclip_cxr_518.json" #Set to biomedclip_cxr_518.json to use the Huggingface default
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


################## Data ##################
#data_path=/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/scripts/data.jsonl
data_path=/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/data/chexpert/processed/llavarad_train.jsonl
image_folder=/data/raw_data/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0
loader="chexpert_train_findings_impressions"
################## Data ##################

################## Run name ##################
epoch="${2:-3}"
bsz="${3:-16}"
lr="1e-5"
master_port="${MASTER_PORT:-29601}"
schedule="lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################

# ''' Arguments
# model_base: Base-model to be fine-tuned, set to vicuna-7b-v1.5 by default.
# data_path: Path to the training data JSON file. Set to /home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/scripts/data.jsonl
# loader: Data loader type, set to chexpert_train_findings_impressions for the chexpert dataset.
# image_folder: Path to the folder containing images. Set to /data/raw_data/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/
# vision_tower: Vision tower model, Use the vision folder as is to use the internal vision tower model.
# vision_tower_config: Remove when using open source vision model from HuggingFace. Keep as is for the internal model.
# vision_tower_checkpoint: Remove when using open source vision model from HuggingFace. Keep as is for the internal model.
# PROJECTOR: Path to the pretrained mm_projector generated from pretrain.sh. It doesn't look like we use the pretrained projector here.
# output_dir: Directory to save the fine-tuned model checkpoints.
# epoch: Number of training epochs, set to 3 by default.
# bsz: Batch size per GPU, set to 16 by default.
# lr: Learning rate, set to 1e-4 by default.
# run_name: Name of the training run, automatically generated based on parameters and timestamp.
# dataloader_num_workers: Change as per number of GPUs available. Set to 4 for the new server.
# '''

# Batch size is set for 4-GPU machines.
# Log checkpoints as W&B model artifacts so they can be downloaded later.
WANDB_PROJECT="llava-rad-finetuning" WANDB_RUN_ID="llava-ft-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=fine-tune WANDB_LOG_MODEL=checkpoint \
    deepspeed --master_port ${master_port} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --pretrain_mm_mlp_adapter ${PROJECTOR} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name ${run_name}
