import sys
sys.path.append("..")

from segment_anything import sam_model_registry


def load_medsam(checkpoint_path, device, eval=True):
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    if eval:
        medsam_model.eval();
    return medsam_model