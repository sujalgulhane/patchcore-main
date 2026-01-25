import argparse
import os
from pathlib import Path
import cv2
from PIL import Image
import torch

from models.patch_core import PatchCore
from models.patch_core import visualize


def run_image(input_path: str, weights_path: str | None = None, out_dir: str | None = None, show: bool = False, device: str | None = None):
    input_path = str(input_path)
    if weights_path is None:
        default_weights = os.path.join('data', 'weights', 'wide_resnet50_size224_param_0.1_9_wood.pth')
        if os.path.exists(default_weights):
            weights_path = default_weights
        else:
            raise FileNotFoundError('weights_path not provided and default weights not found')

    # Load model with weights (loads memory bank etc.)
    net = PatchCore.load_weights(weights_path, device)

    # Read original image (BGR)
    im_org = cv2.imread(input_path)
    if im_org is None:
        raise FileNotFoundError(f'Input image not found: {input_path}')

    # Prepare tensor input
    im_pil = Image.open(input_path).convert('RGB')
    x = net.get_transform()(im_pil)
    x = net.get_resize()(x)
    x = torch.unsqueeze(x, 0).to(net.device)

    # Predict
    anomaly_score, anomaly_map, pred = net.predict(x, th=0.5)

    # Create heatmap and overlay
    im_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)
    im_add = visualize.add_image(im_heatmap, im_org, alpha=0.5)

    # Save outputs only if out_dir provided
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        base = Path(input_path).stem
        heat_path = os.path.join(out_dir, f"{base}_heatmap.png")
        add_path = os.path.join(out_dir, f"{base}_overlay.png")
        cv2.imwrite(heat_path, im_heatmap)
        cv2.imwrite(add_path, im_add)
    else:
        heat_path = None
        add_path = None

    print(f"anomaly_score: {float(anomaly_score):.6f}")
    print(f"pred: {pred}")
    if out_dir is not None:
        print(f"Saved heatmap: {heat_path}")
        print(f"Saved overlay: {add_path}")
    else:
        print("Outputs were not saved (no output directory specified)")
    if show:
        # Windows: open with default viewer
        try:
            os.startfile(add_path)
        except Exception:
            print('Opening image is not supported on this platform')

    return anomaly_score, pred, heat_path, add_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='path to input image')
    parser.add_argument('--weights', '-w', help='weights path', default=None)
    parser.add_argument('--out', '-o', help='output dir', default=None)
    parser.add_argument('--show', action='store_true', help='open overlay image with default viewer (Windows only)')
    args = parser.parse_args()

    run_image(args.input_path, args.weights, args.out, args.show)
