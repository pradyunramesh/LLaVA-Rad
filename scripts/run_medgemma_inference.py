"""
MedGemma inference script for CheXpert evaluation.
Generates predictions.jsonl compatible with LLaVA-Rad evaluation pipeline.
"""
import os
import json
import torch
import fire
from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoModelForCausalLM
import math

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_medgemma(
    query_file: str,
    image_folder: str,
    prediction_file: str,
    model_path: str = "google/medgemma-1.5-4b-it",
    device: str = "cuda",
    temperature: float = 0.0,
    top_p: float = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    chunk_idx: int = 0,
    num_chunks: int = 1,
    batch_size: int = 1,  # MedGemma may need batch_size=1 for vision models
):
    """
    Run MedGemma inference on CheXpert query file.
    
    Args:
        query_file: Path to chexpert_query.jsonl
        image_folder: Base folder containing CheXpert images
        prediction_file: Output path for predictions.jsonl
        model_path: HuggingFace model path for MedGemma
        device: Device to run on (cuda/cpu)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum tokens to generate
        chunk_idx: Chunk index for parallel processing
        num_chunks: Total number of chunks
        batch_size: Batch size (typically 1 for vision-language models)
    """
    print(f"Loading MedGemma model from: {model_path}")
    print(f"Device: {device}")
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    # Load queries
    print(f"Loading queries from: {query_file}")
    all_queries = []
    with open(query_file, 'r') as f:
        for line in f:
            if line.strip():
                query = json.loads(line)
                # Only process validation split for evaluation
                if query.get("split") == "valid":
                    all_queries.append(query)
    
    print(f"Filtered to {len(all_queries)} validation queries")
    queries = get_chunk(all_queries, num_chunks, chunk_idx)
    print(f"Processing {len(queries)} / {len(all_queries)} queries (chunk {chunk_idx}/{num_chunks})")
    
    # Create output directory
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    
    # Process queries
    with open(prediction_file, 'w') as pred_file:
        for query in tqdm(queries, desc=f"Chunk {chunk_idx}"):
            try:
                # Extract query information
                image_path = query["image"]
                full_image_path = os.path.join(image_folder, image_path)
                
                # Get prompt from conversations
                # Extract the text part (remove <image> token)
                prompt_text_raw = query["conversations"][0]["value"]
                # Remove <image> token and clean up
                prompt_text_raw = prompt_text_raw.replace("<image>", "").strip()
                if prompt_text_raw.startswith("\n"):
                    prompt_text_raw = prompt_text_raw[1:].strip()
                
                # Get reference (ground truth)
                reference = query["conversations"][1]["value"] if len(query["conversations"]) > 1 else ""
                
                # Load and process image
                if not os.path.exists(full_image_path):
                    print(f"Warning: Image not found: {full_image_path}")
                    continue
                
                image = Image.open(full_image_path).convert("RGB")
                
                # MedGemma expects chat template format with image tokens
                # Format: messages with role 'user' and content containing image and text
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text_raw}
                        ]
                    }
                ]
                
                # Apply chat template to get the formatted prompt
                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Process inputs - processor expects text with image tokens and images
                inputs = processor(images=[image], text=prompt_text, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Generate
                with torch.inference_mode():
                    generate_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": temperature > 0,
                        "temperature": temperature if temperature > 0 else None,
                        "top_p": top_p if top_p else None,
                        "num_beams": num_beams,
                    }
                    # Remove None values
                    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
                    
                    outputs = model.generate(**inputs, **generate_kwargs)
                
                # Decode output
                prediction = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (remove input prompt)
                # MedGemma chat template format can include various markers
                # Strategy: Find the last occurrence of key markers and extract everything after
                
                # First, try to find the model response marker
                markers = [
                    "<start_of_turn>model\n",
                    "<start_of_turn>model",
                    "model\n",
                    "\nmodel\n",
                ]
                
                prediction_cleaned = None
                for marker in markers:
                    if marker in prediction:
                        parts = prediction.split(marker)
                        if len(parts) > 1:
                            prediction_cleaned = parts[-1].strip()
                            break
                
                if prediction_cleaned:
                    prediction = prediction_cleaned
                else:
                    # Fallback: try to remove the prompt text
                    if prompt_text in prediction:
                        prediction = prediction.split(prompt_text)[-1].strip()
                
                # Remove any remaining special tokens
                prediction = prediction.replace("<end_of_turn>", "").strip()
                prediction = prediction.replace("<start_of_turn>", "").strip()
                
                # Remove any leading "user" or "model" text patterns
                # Handle cases like "user\n\n\n\n\n..." or "model\nThe..."
                import re
                prediction = re.sub(r'^user\s*\n+', '', prediction, flags=re.MULTILINE)
                prediction = re.sub(r'^model\s*\n+', '', prediction, flags=re.MULTILINE)
                prediction = prediction.strip()
                
                # If prediction still starts with the query text, remove it
                if prediction.startswith(prompt_text_raw):
                    prediction = prediction[len(prompt_text_raw):].strip()
                
                # Write prediction in LLaVA-Rad format
                # Use the original prompt text (without chat template formatting) for the query field
                output_record = {
                    "image": image_path,
                    "query": prompt_text_raw,  # Use original prompt text
                    "reference": reference,
                    "prediction": prediction,
                    "generation loss": 0.0  # MedGemma doesn't compute loss during generation
                }
                
                pred_file.write(json.dumps(output_record) + "\n")
                pred_file.flush()
                
            except Exception as e:
                print(f"Error processing {query.get('image', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"Predictions written to: {prediction_file}")


if __name__ == "__main__":
    fire.Fire(eval_medgemma)
