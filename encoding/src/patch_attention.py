import os
from typing import List
from pathlib import Path

import cv2
import timm
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from utils import (
    PatchingDataset,
    get_args,
    get_model
)


def create_attention_maps(
    model: timm.models.vision_transformer.VisionTransformer,
    dataloader: PatchingDataset,
    device: str,
    save_dir: str
    ) -> torch.Tensor:

    attention_maps = []

    def hook_fn(module, input, _):
        batch_size, num_tokens, embed_dim = input[0].shape
        qkv = module.qkv(input[0])
        qkv = qkv.reshape(batch_size, num_tokens, 3, module.num_heads, embed_dim // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]

        attention = torch.matmul(q, k.transpose(-2, -1)) * module.scale
        attention = attention.softmax(dim=-1)

        attention_maps.append(attention.cpu())

    model.eval()
    for block in model.blocks:
        block.attn.register_forward_hook(hook_fn)

    with torch.inference_mode():
        for img, coords, img_path in tqdm(dataloader, desc="Generating Attention Maps"):
            img = img.to(device)
            patch_path = os.path.join(save_dir, f"{coords[0]}.png")

            _ = model(img)

            final_block = attention_maps[-1]
            cls_attention = final_block[0, :, 0, :]
            cls_avg = cls_attention.mean(dim=0)[1:]
            dim = int(np.sqrt(cls_attention.shape[-1]))

            attention_map = cls_avg.reshape(dim, dim).numpy()
            
            patch_size = 16
            attention_map_upscaled = np.kron(attention_map, np.ones((patch_size, patch_size)))
            attention_map_upscaled = (attention_map_upscaled - attention_map_upscaled.min()) / (attention_map_upscaled.max() - attention_map_upscaled.min())

            attention_bgr = cv2.applyColorMap((attention_map_upscaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            img = cv2.imread(img_path[0])

            alpha = 0.5
            overlay = cv2.addWeighted(img, 0.5, attention_bgr, alpha, 0)
            cv2.imwrite(patch_path, overlay)
            attention_maps = []

def main():
    load_dotenv(os.path.join("..", ".env"))
    hf_token = os.getenv("HF_TOKEN")

    arg_dir = os.path.join("configs", "attention-config.yaml")
    args = get_args(arg_dir)

    data_dir = os.path.join("..", "..", "raw-data", "patches", f"experiment-{args['experiment_num']}")
    dest_dir = os.path.join("..", "..", "raw-data", "attention-maps", f"experiment-{args['experiment_num']}", args["model"])

    os.makedirs(dest_dir, exist_ok=True)
    selected_ids = list(map(str, [patient_id for patient_id in args["selected_patients"]]))

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps"
    model = get_model(args["model"], device, hf_token).to(device)

    for i, patient_id in enumerate(selected_ids):
        print(f"patient: {patient_id} | [{i+1}/{len(selected_ids)}]")

        id_dir = os.path.join(data_dir, patient_id)
        save_dir = os.path.join(dest_dir, patient_id)
        os.makedirs(save_dir, exist_ok=True)

        patch_dataset = PatchingDataset(id_dir, return_img_path=True)
        patch_loader = DataLoader(patch_dataset, batch_size=1)
        create_attention_maps(model, patch_loader, device, save_dir)

        print(f"\nAttention maps saved")
        print("\n-------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()