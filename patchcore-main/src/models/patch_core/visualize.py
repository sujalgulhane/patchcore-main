from __future__ import annotations

import numpy as np
import cv2
from copy import deepcopy
import torch
import colorsys

def color_map() -> np.array:
    """Generate color map.

    Returns:
        np.array: Color map
    """
    b = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.0, 1/6, 103)]
    g = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(1/6, 1/3, 25)]
    r = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.5, 2/3, 128)]

    rgb = np.asarray((np.array(b + g + r) * 255), dtype=np.uint8)

    return rgb

def color_map2() -> np.array:
    """Generate color map (unused).

    Returns:
        np.array: Color map
    """
    
    rgb = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.0, 0.6666666666, 256)]

    rgb = np.asarray((np.array(rgb) * 255), dtype=np.uint8)

    return rgb

def create_color_bar_image(color_map_func=color_map, save_path="color_bar.png", w=5, h=80):
    """Generate and save a color bar image for heatmaps.

    Args:
        color_map_func: Function to generate color map. Defaults to color_map.
        save_path (str, optional): Path to save the image. Defaults to "color_bar.png".
        w (int, optional): Width in pixels per color block. Defaults to 5.
        h (int, optional): Height in pixels per color block. Defaults to 80.
    """
    cm = color_map_func()

    bar = np.zeros((h, 256*w, 3), dtype=np.uint8)
    for i in range(256):
        bar[:, i*w:i*w+w] = cm[i]

    cv2.imwrite(save_path, bar) 

def create_heatmap_image(
        anomaly_map: torch.Tensor,
        org_size: tuple[int, int] | None=None,
    ) -> np.array:
    """Generate a heatmap image from an anomaly map.

    Args:
        anomaly_map (torch.Tensor): Anomaly map tensor.
        org_size (tuple[int, int] | None, optional): Original image size (height, width). Defaults to None.

    Returns:
        np.array: Heatmap image
    """
    anomaly_map = anomaly_map.detach().cpu().numpy()

    map = anomaly_map[0][0]

    # Resize to specified size
    if org_size is not None:
        map = cv2.resize(map, (org_size[1], org_size[0]))

    map = (map * 255).astype(np.uint8)

    new_map = np.take(color_map(), map, axis=0)

    return new_map

def add_image(
        im_heatmap: np.array,
        im_org: np.array,
        alpha=0.3,
    ) -> np.array:
    """Combine heatmap image and original image.

    Args:
        im_heatmap (np.array): Heatmap image.
        im_org (np.array): Original image.
        alpha (float, optional): Alpha blending factor. Defaults to 0.3.

    Returns:
        np.array: Blended image
    """
    im_heatmap = cv2.resize(im_heatmap, (im_org.shape[1], im_org.shape[0]), interpolation=cv2.INTER_CUBIC)
    im_mask = _create_mask(im_heatmap)
    im_org = cv2.bitwise_or(im_org, im_org, mask=im_mask)

    im_add = cv2.addWeighted(src1=im_heatmap, alpha=alpha, src2=im_org, beta=1-alpha, gamma=0)

    return im_add

def _create_mask(im: np.array) -> np.array:
    """Create mask image.

    Args:
        im (np.array): Input image.

    Returns:
        np.array: Mask image
    """
    im_mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_mask = cv2.bitwise_not(im_mask)
    white = np.full(im_mask.shape, 255, dtype=np.uint8)
    im_mask = cv2.bitwise_or(white, white, mask=im_mask)

    return im_mask
