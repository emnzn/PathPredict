import os
import timm
import torch
from huggingface_hub import hf_hub_download

def get_model(
    model: str, 
    device: str, 
    token: str
    ) -> timm.models.vision_transformer.VisionTransformer:

    if model == "UNI":
        model_dir = os.path.join("..", "assets", "pretrained-weights", "UNI-mass100k")
        filename="pytorch_model.bin"
        filepath = os.path.join(model_dir, filename)

        os.makedirs(model_dir, exist_ok=True)

        if not os.path.isfile(filepath):
            hf_hub_download("MahmoodLab/UNI", filename=filename, local_dir=model_dir, force_download=True, token=token)
        
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        state_dict = torch.load(filepath, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
    return model
