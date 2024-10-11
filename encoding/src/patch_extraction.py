import os
from pathlib import Path

import pyvips

from utils import (
    pad_img,
    get_args, 
    patchify,
    save_patches,
    save_patch_args
    )


def main():
    data_dir = os.path.join("..", "..", "raw-data", "images")
    arg_dir = os.path.join("configs", "patch-config.yaml")
    patch_dir = os.path.join("..", "..", "raw-data", "patches")
    
    os.makedirs(patch_dir, exist_ok=True)

    experiments  = os.listdir(patch_dir)
    experiment_num = len(experiments)
    output_dir = os.path.join(patch_dir, f"experiment-{experiment_num}")
    
    args = get_args(arg_dir)
    save_patch_args(output_dir, args)
    image_names = os.listdir(data_dir)
    image_paths = [os.path.join(data_dir, i) for i in image_names]

    for i, image_path in enumerate(image_paths):
        image_name = Path(image_path).stem.split(".")[0]

        print(f"Patient-{image_name} [{i+1}/{len(image_paths)}]")
        wsi = pyvips.Image.new_from_file(image_path, access="sequential").numpy()

        target_shape = args["target_shape"]
        
        if target_shape != wsi.shape[:2]: 
            wsi = pad_img(wsi, target_shape) 

        image_patches, coordinates = patchify(wsi, args["patch_size"], args["overlap"])

        save_patches(
            image_patches, 
            coordinates, 
            os.path.join(output_dir, image_name)
        )

        print("\n---------------------------------------------\n")
        

if __name__ == "__main__":
    main()