from typing import List

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import transform

from experiments.constants import LUNG_WIN, ABD_WIN, BONE_WIN, HU_FACTOR, ALL_WIN
from experiments.viz_utils import plot_results


def slice_num(filename):
    return int(filename.split(".")[0])


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
    bone = apply_window(ct_slice, BONE_WIN['center'], BONE_WIN['width'])
    return lung, abdomen, bone


def to_uint8(arr):
    if arr.max() <= 1:
        arr *= 255
    return np.array(arr, dtype=np.uint8)


def process_img_array(img_hu: np.ndarray, rgb=False):
    lung, abdomen, bone = window_ct(img_hu)
    return lung, abdomen, bone

def process_slice(slice_path, rgb=False):
    img = Image.open(slice_path)
    return process_img_array(img)


def process_bbox_str(bbox):
    return [float(x) for x in bbox.split(", ")]


# ------------------------- MedSAM Utility Functions -------------------------

def transform_img(img_3c, device):
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


def segment(img, bbox, model):
    # img_np = np.array(img.convert('RGB'))
    img_np = np.array(img) if type(img) is Image.Image else img
    if len(img_np.shape) == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
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
    return [xmin, ymin, xmax, ymax]


def add_margin_to_bbox(bbox, margin):
    x_min, y_min, x_max, y_max = bbox
    return [x_min - margin, y_min - margin, x_max + margin, y_max + margin]


def expand_to_square_bbox(bbox, ratio, min_size=None):
    """
    Convert bbox to square w longest side, then expand it by a ratio
    """
    x_min, y_min, x_max, y_max = bbox
    side = max(x_max - x_min, y_max - y_min) * ratio
    if min_size:
        side = max(side, min_size)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_side = side / 2
    return [cx - half_side, cy - half_side, cx + half_side, cy + half_side]


def segment_slices(
        model, slices, start_box, slice_group, slice_indices, window,
        plot=False, save=False, margin_ratio=0.1
):
    # the order of slice matters - start with key slice then extend out
    # area to show
    zoom_box = expand_to_square_bbox(start_box, 3, min_size=200)
    slice_segs = []
    bbox = start_box.copy()

    for slice_img, i in zip(slices, slice_indices):
        slice_arr = np.array(slice_img, dtype=int) - HU_FACTOR
        if window:
            lung, abdomen, bone = window_ct(slice_arr)
            slice_arr = abdomen
        else:
            slice_arr = apply_window(slice_arr, ALL_WIN['center'], ALL_WIN['width'])

        seg = segment(slice_arr, bbox, model)
        segs = split_seg(seg)

        out_name = f"start{slice_indices[0]}_{i}"
        save_path = f"outputs/extend_3d/{slice_group}/{out_name}" if save else None
        plot_results(slice_arr, [bbox], segs, zoom_box, plot=plot, save_path=save_path)
        # update bbox
        if segs[0].sum() == 0:
            break
        bbox = get_seg_bbox(segs[0])
        margin = round(margin_ratio * max(bbox[2] - bbox[0], bbox[3] - bbox[1])) + 1
        bbox = add_margin_to_bbox(bbox, margin)
        slice_segs.append(seg)
        break
    return slice_segs


def generate_seg(img_path, seg_path, medsam) -> np.ndarray:
    # load ct and segmentation
    ct = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(seg)

    # get largest slice to be the key slice
    seg_slice_size = seg_array.sum(axis=(1, 2))
    seg_indices = np.nonzero(seg_slice_size)[0]
    key_index = seg_slice_size.argmax()

    ## Note: Only run SAM on slices with segments i.e. doesn't test how SAM stop segmenting

    # get sequence of slices indices for up and down
    up_indices = [i for i in seg_indices if i > key_index]
    down_indices = [i for i in seg_indices if i < key_index][::-1]

    # get bbox from GT seg
    start_box = get_seg_bbox(seg_array[key_index])

    # segment key slice, then up and down
    seg_key = segment_slices(medsam, ct_array[[key_index]], [key_index], start_box)
    segs_up = segment_slices(medsam, ct_array[up_indices], up_indices, start_box)
    segs_down = segment_slices(medsam, ct_array[down_indices + [63]], down_indices + [63], start_box)

    # insert seg into an empty array size of original CT
    full_vol_seg = np.zeros(ct_array.shape)
    vol_seg = np.stack(segs_down[::-1] + seg_key + segs_up[1:])
    for idx, slice in zip(seg_indices, vol_seg):
        full_vol_seg[idx] = slice

    # binary mask
    return (full_vol_seg > 0.5).astype(np.uint8)
