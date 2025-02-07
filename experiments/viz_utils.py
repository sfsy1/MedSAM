from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np


def transparent_cmap(color):
    ncolors = 256
    color_array = plt.get_cmap('Greys')(range(ncolors))
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    color_array[:, 0:3] = to_rgb(color)
    return LinearSegmentedColormap.from_list(name=f'{color}_transparent', colors=color_array)



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="cyan", facecolor=(0, 0, 0, 0), lw=0.5)
    )


def show_mask(mask, ax, color=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if color is None:
            color = np.array([30 / 255, 30 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_results(img, box, seg, zoom_box, plot=True, save_path=False):
    fig, ax = plt.subplots(1, 2, figsize=(5, 3))
    fig.subplots_adjust(wspace=0.02, hspace=0, left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("k")
    ax[0].imshow(img, cmap="gray")
    if box:
        show_box(box[0], ax[0])
    ax[0].set_title("bbox prompt", fontdict={'color': 'white'})

    ax[1].imshow(img, cmap="gray")
    seg_alpha = 0.4
    if seg is not None:
        if type(seg) is np.ndarray:
            show_mask(seg, ax[1], color=np.array([0.1, 0.1, 1, seg_alpha]))
        else:
            seg_low, seg_mid, seg_high = seg

            show_mask(seg_low, ax[1], color=np.array([1, 0.1, 0.1, seg_alpha]))
            show_mask(seg_mid, ax[1], color=np.array([1, 1, 0.1, seg_alpha]))
            show_mask(seg_high, ax[1], color=np.array([0.1, 1, 0.1, seg_alpha]))

    if box:
        show_box(box[0], ax[1])
    ax[1].set_title("segmentation", fontdict={'color': 'white'})

    if zoom_box:
        # zoom into the crop
        for i in range(len(ax)):
            ax[i].set_xlim(zoom_box[0], zoom_box[2])
            ax[i].set_ylim(zoom_box[1], zoom_box[3])

    for ax in ax:
        ax.axis('off')
    if plot:
        plt.show()
    elif save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_path}.jpg", bbox_inches='tight', pad_inches=0, dpi=180)
        plt.close(fig)
    else:
        plt.close(fig)