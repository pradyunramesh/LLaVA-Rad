#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

model_base=lmsys/vicuna-7b-v1.5
output_dir="${1:-./checkpoints}"

# data_path=/PATH_TO/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json

loader="mimic_train_findings"

# image_folder=/PATH_TO/physionet.org/files/mimic-cxr-jpg/2.0.0/files

''' Arguments
model_base: Base-model to be fine-tuned, set to vicuna-7b-v1.5 by default.
data_path: Path to the training data JSON file. Set to /home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/scripts/data.jsonl
loader: Data loader type, set to chexpert_train_findings_impressions for the chexpert dataset.
image_folder: Path to the folder containing images. Set to /data/raw_data/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/
vision_tower: Vision tower model, set to hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224. Use the vision folder as is to use the internal vision tower model.
vision_tower_config: Remove when using open source vision model from HuggingFace. Keep as is for the internal model.
vision_tower_checkpoint: Remove when using open source vision model from HuggingFace. Keep as is for the internal model.
output_dir: Directory to save the fine-tuned model checkpoints. Set by default to the ./checkpoints folder in the current directory.
epoch: Number of training epochs, set to 3 by default.
bsz: Batch size per GPU, set to 16 by default.
lr: Learning rate, set to 1e-4 by default.
grad_acc: Gradient accumulation steps, set to 2 by default. Model weights are updated after every other batch.
run_name: Name of the training run, automatically generated based on parameters and timestamp.
dataloader_num_workers: Change as per number of GPUs available. Set to 4 for the new server.
'''

################## Run name ##################
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt" 

epoch="${2:-1}"
bsz="${3:-32}"
grad_acc="${4:-2}"
lr="1e-3"
schedule="pt-${epoch}e"
run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################

# Global batch size should be 256

WANDB_RUN_ID="llava-pt-$(date +%Y%m%d%H%M%S)" WANDB_PROJECT="llava" WANDB_RUN_GROUP=pre-train \
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${model_base} \
    --version plain \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${grad_acc} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${run_name}
