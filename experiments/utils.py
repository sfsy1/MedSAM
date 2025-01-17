from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import transform
from typing import List

from experiments.constants import LUNG_WIN, ABD_WIN, HU_FACTOR
from experiments.viz_utils import plot_results


def load_process_csv(csv_path):
    df = pd.read_csv(csv_path)

    # get file/path info
    df['scan_name'] = df['File_name'].str.split("_").str[:-1].str.join("_")
    df['file_name'] = df['File_name'].str.split("_").str[-1]
    df['image_path'] = df['scan_name'] + "/" + df['file_name']

    # aspect ratio of voxel (vertical / traverse pixel spacing)
    spacing = df.Spacing_mm_px_.str.split(", ")
    df['aspect_ratio'] = spacing.str[2].astype(float) / spacing.str[0].astype(float)
    df['bbox'] = df['Bounding_boxes'].str.split(", ").apply(lambda x: [float(y) for y in x])
    return df


def get_slices_filenames(slice_range: list):
    min_id, max_id = [int(x) for x in slice_range]
    return [f"{str(i).zfill(3)}.png" for i in range(min_id, max_id + 1)]


def clip_normalize(img_arr, min_value=None, max_value=None, remove_high=True):
    """
    remove_high: values > max_value is set to min_value i.e. 0 after normalization
    """
    img_arr = np.array(img_arr, dtype=float)
    if min_value is None:
        min_value = img_arr.min()
    if max_value is None:
        max_value = img_arr.max()
    img_arr = img_arr.copy()
    if remove_high:
        img_arr[img_arr > max_value] = min_value
    img_arr = img_arr.clip(min_value, max_value)
    return (img_arr - min_value) / (max_value - min_value)


def apply_window(image, center, width):
    window_min = center - width // 2
    window_max = center + width // 2
    return np.clip(image, window_min, window_max)


def window_ct(ct_slice):
    lung = apply_window(ct_slice, LUNG_WIN['center'], LUNG_WIN['width'])
    abdomen = apply_window(ct_slice, ABD_WIN['center'], ABD_WIN['width'])
    return lung, abdomen


def to_uint8(arr):
    if arr.max() <= 1:
        arr *= 255
    return np.array(arr, dtype=np.uint8)


def process_slice(slice_path, rgb=False):
    img = Image.open(slice_path)
    img_hu = np.array(img, dtype=int) - HU_FACTOR
    lung, abdomen = window_ct(img_hu)
    lung_img = Image.fromarray(to_uint8(clip_normalize(lung)))
    abdomen_img = Image.fromarray(to_uint8(clip_normalize(abdomen)))
    if rgb:
        lung_img = lung_img.convert('RGB')
        abdomen_img = abdomen_img.convert('RGB')
    return lung_img, abdomen_img


def process_bbox_str(bbox):
    return [float(x) for x in bbox.split(", ")]


# ------------------------- MedSAM Utility Functions -------------------------

def transform_img(img_np, device):
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    return torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False
    )
    return low_res_pred.squeeze().cpu().numpy()


def segment(img: Image, bbox, model):
    img_np = np.array(img.convert('RGB'))
    in_tensor = transform_img(img_np, model.device)

    with torch.no_grad():
        image_embedding = model.image_encoder(in_tensor)  # (1, 256, 64, 64)

    H, W, _ = img_np.shape
    box_np = np.array([bbox])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    return medsam_inference(model, image_embedding, box_1024, H, W)


def split_seg(seg, bins=[0.1, 0.4, 0.7, 1]) -> List[np.ndarray]:
    splits = []
    for left, right in zip(bins[:-1], bins[1:]):
        splits.append(((seg > left) & (seg <= right)).astype(np.uint8))
    return splits


def get_seg_bbox(seg):
    mask = np.asarray(seg)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax


def add_margin_to_bbox(bbox, margin):
    x_min, y_min, x_max, y_max = bbox
    return [x_min - margin, y_min - margin, x_max + margin, y_max + margin]


def expand_to_square_bbox(bbox, ratio):
    """
    Convert bbox to square w longest side, then expand it by a ratio
    """
    x_min, y_min, x_max, y_max = bbox
    side = max(x_max - x_min, y_max - y_min) * ratio
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_side = side / 2
    return [cx - half_side, cy - half_side, cx + half_side, cy + half_side]



def segment_slice_sequence(model, slice_paths, start_box, plot, save=None):
    # the order of slice matters - start with key slice then extend out
    zoom_box = expand_to_square_bbox(start_box, 4)
    slices = []
    slice_segs = []
    bbox = start_box.copy()
    first_slice = int(slice_paths[0].stem.split(".")[0])
    for i, slice_path in enumerate(slice_paths):
        lung, abdomen = process_slice(slice_path, rgb=True)
        seg = segment(abdomen, bbox, model)
        segs = split_seg(seg)

        slice_id = int(slice_path.stem.split(".")[0])
        out_name = f"start{first_slice}_{slice_id}"
        save_path = f"outputs/extend_3d/{slice_path.parent.stem}/{out_name}" if save else None

        plot_results(abdomen, [bbox], segs, zoom_box, plot=plot, save_path=save_path)

        # update bbox
        bbox = get_seg_bbox(segs[0])
        margin = round(0.1 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        bbox = add_margin_to_bbox(bbox, margin)
        slices.append(abdomen)
        slice_segs.append(seg)

    return slices, slice_segs