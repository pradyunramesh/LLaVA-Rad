#!/bin/bash

set -e
set -o pipefail

# Set Hugging Face cache to a location with more disk space
export HF_HOME=/lfs/skampere2/0/mahmedc/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Default paths
model_path="${1:-google/medgemma-1.5-4b-it}"
prediction_dir="${2:-evaluation_results/medgemma_base}"
prediction_file=$prediction_dir/predictions

query_file=/lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/scripts/chexpert_query.jsonl
image_folder=/lfs/skampere2/0/mahmedc/CUMC_Radiology/data/raw_data/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0

# Number of GPUs to use (set to 1 for now, can be increased if needed)
CHUNKS=1

echo "Running MedGemma inference..."
echo "Model: $model_path"
echo "Query file: $query_file"
echo "Image folder: $image_folder"
echo "Output: $prediction_file"

for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=$idx python scripts/run_medgemma_inference.py \
        --query_file ${query_file} \
        --image_folder ${image_folder} \
        --prediction_file ${prediction_file}_${idx}.jsonl \
        --model_path ${model_path} \
        --temperature 0 \
        --num_beams 1 \
        --max_new_tokens 512 \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size 1 &
done

wait

# Combine chunks if multiple GPUs were used
if [ $CHUNKS -gt 1 ]; then
    cat ${prediction_file}_*.jsonl > ${prediction_file}.jsonl
    rm ${prediction_file}_*.jsonl
else
    mv ${prediction_file}_0.jsonl ${prediction_file}.jsonl
fi

echo "Predictions saved to: ${prediction_file}.jsonl"
