import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from empatches import EMPatches


def patchify(
    wsi: np.ndarray,
    patchsize: int,
    overlap: float
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:

    """
    Patchify a whole slide image (WSI) into smaller patches.

    Parameters
    ----------
    wsi: np.ndarray
        The whole slide image as a numpy array.

    patchsize: int
        The size of each patch.

    overlap: float
        The overlap between patches.

    Returns
    -------
    image_patches: List[np.ndarray]
        A list of image patches.

    coordinates: List[Tuple[int, int, int, int]]
        A list of tuples containing the coordinates of each patch.
    """

    emp = EMPatches()
    image_patches, coordinates = emp.extract_patches(wsi, patchsize=patchsize, overlap=overlap)

    return image_patches, coordinates


def save_patches(
    image_patches: List[np.ndarray],
    coordinates: List[Tuple[int, int, int, int]],
    output_dir: str,
    ) -> None:

    """
    Save the image patches as individual images.

    Parameters
    ----------
    image_patches: List[np.ndarray]
        A list of image patches.

    indices: List[Tuple[int, int, int, int]]
        A list of tuples containing the coordinates of each patch.

    output_dir: str
        The directory where the image patches will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm(zip(image_patches, coordinates), desc="Patching in progress", total=len(coordinates))

    for patch, coordinates in pbar:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        is_valid = valid_patch(patch)

        if is_valid:
            y1, y2, x1, x2 = coordinates

            patch_name = f"patch-{y1}-{y2}-{x1}-{x2}.png"
            patch_path = os.path.join(output_dir, patch_name)

            cv2.imwrite(patch_path, patch, [cv2.IMWRITE_PNG_COMPRESSION, 5])


def extract_coords(img_name: List[str]) -> Tuple[int]:

    """
    Extracts the coordinates of a given patch from its filename.
    """

    stem = Path(img_name).stem
    y1, y2, x1, x2 = [int(c) for c in stem.split("-")[1:]]
    coords = (y1, y2, x1, x2)

    return coords


def get_nearest_multiple(source, target) -> int:

    """
    Calculate the nearest multiple of a target number to a given source number.

    Parameters
    ----------
    source : int
        The number from which to find the nearest multiple.

    target : int
        The number whose multiple is to be found.

    Returns
    -------
    nearest_multiple: int
        The nearest multiple of `target` to `source`. If `source` is already a multiple of `target`, it returns `source`.
    
    Notes
    -----
    This function rounds up to the next multiple of `target` if `source` is not already a multiple of `target`.
    """
    
    remainder = source % target

    nearest_multiple = (source + (target - remainder)) if remainder else source
    
    return nearest_multiple


def get_target_shape(img: np.ndarray, patch_size: int) -> Tuple[int]:
    """
    Calculates the target shape of an image, to become a multiple of the patch size.

    Returns
    -------
    target_size: Tuple[int]
        The size the image should be to become a multiple of the target number.
    """

    source_height, source_width = img.shape[0], img.shape[1]

    target_height = get_nearest_multiple(source_height, patch_size)
    target_width = get_nearest_multiple(source_width, patch_size)

    target_size = (target_height, target_width)

    return target_size


def pad_img(img: np.ndarray, target_shape: Tuple[int]) -> np.ndarray:

    """
    Pads the image to a target shape.
    """

    current_shape = img.shape[:2]

    delta_h = target_shape[0] - current_shape[0]
    delta_w = target_shape[1] - current_shape[1]

    pad_top = delta_h // 2
    pad_bottom = delta_h - pad_top
    
    pad_left = delta_w // 2
    pad_right = delta_w - pad_left

    padded_img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, 
        pad_left, pad_right, 
        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    return padded_img


def valid_patch(img: np.ndarray, threshold: int = 230) -> bool:

    """
    Checks whether a patch is mostly background.
    Returns false if the patch contains 75% or more background pixels.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, background_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    white_pixels = np.sum(background_mask == 255)
    total_pixels = img.shape[0] * img.shape[1]

    background_composition = white_pixels / total_pixels

    is_valid = bool(background_composition < 0.75)

    return is_valid
    