import os
import torch
import random
import numpy as np
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)  # Enable deterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Set CUDA environment for determinism

# Ensure CPU operations are also deterministic
random.seed(42)
np.random.seed(42)

model_path = "/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/checkpoints/biomedclip_cxr_518-lora-3e-1e-4-20250907223658"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base="lmsys/vicuna-7b-v1.5", model_name=model_name, load_8bit = False, load_4bit = False, device="cuda"
)

image = Image.open("/data/raw_data/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train/patient53157/study1/view1_frontal.jpg").convert('RGB')
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
print(image_tensor.shape)

with torch.inference_mode():
    vision_tower = model.get_vision_tower()
    projected_image_features = model.encode_images(image_tensor.unsqueeze(0).cuda().half())
    raw_features = vision_tower(image_tensor.unsqueeze(0).cuda().half())
    projected_features = model.get_model().mm_projector(raw_features)
    print("Raw image features shape:", raw_features.shape)
    print("Projected image features shape:", projected_image_features.shape)
    
    # Check if tensors are equal using torch.allclose
    # This handles small floating point differences with a tolerance
    tensors_match = torch.allclose(projected_image_features, projected_features, rtol=1e-5, atol=1e-5)
    print(f"Tensors match: {tensors_match}")
    
    # If you want to see exact differences
    if not tensors_match:
        diff = (projected_image_features - projected_features).abs()
        print(f"Max difference: {diff.max().item()}")
        print(f"Mean difference: {diff.mean().item()}")
