import torch
from huggingface_hub import hf_hub_download

# non_lora_trainables.bin contains the projector and vision tower weights
# from the fine-tuned LLaVA-Rad model
cache_file = hf_hub_download(repo_id="microsoft/llava-rad", filename="non_lora_trainables.bin")
weights = torch.load(cache_file, map_location="cpu")

# Keys are prefixed with "base_model.model." - filter and strip to match mm_projector.bin format
projector_weights = {
    k.replace("base_model.model.", ""): v
    for k, v in weights.items()
    if "mm_projector" in k
}

print(list(projector_weights.keys()))
# Expect:
# ['model.mm_projector.0.weight', 'model.mm_projector.0.bias',
#  'model.mm_projector.2.weight', 'model.mm_projector.2.bias']

torch.save(projector_weights, "/home/pr2762@mc.cumc.columbia.edu/CXR-pipeline/CXR-reason/checkpoints/mm_projector.bin")