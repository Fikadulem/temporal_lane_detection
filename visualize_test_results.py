import argparse
import os

import cv2
import numpy as np
import torch

from model import get_model


def resolve_weights_path(model_type, weights_path=None):
    """Resolve the checkpoint file path for the given model type.

    Uses the explicit path when provided. Falls back to ``lane_model.pth``
    for ConvLSTM or ``lane_model_<model_type>.pth`` for other types.

    Args:
        model_type (str): One of ``'convlstm'`` or ``'cnn'``.
        weights_path (str, optional): Explicit checkpoint path override.

    Returns:
        str: Resolved path to the checkpoint file.
    """
    if weights_path is not None:
        return weights_path

    if model_type == "convlstm" and os.path.exists("lane_model.pth"):
        return "lane_model.pth"

    return f"lane_model_{model_type}.pth"


def resolve_mask_path(mask_dir, frame_name):
    """Find the ground-truth mask file for a given frame filename.

    Tries the exact filename first, then common jpg/png extension swaps.

    Args:
        mask_dir (str): Directory containing mask files.
        frame_name (str): Filename of the corresponding input frame.

    Returns:
        str or None: Full path to the mask file, or ``None`` if not found.
    """
    candidates = [
        frame_name,
        frame_name.replace(".jpg", ".png"),
        frame_name.replace(".png", ".jpg"),
        frame_name.replace(".jpeg", ".png"),
        frame_name.replace(".png", ".jpeg"),
    ]

    for candidate in candidates:
        path = os.path.join(mask_dir, candidate)
        if os.path.exists(path):
            return path

    return None


def sorted_image_paths(folder):
    """Return sorted paths of all image files in a folder.

    Args:
        folder (str): Directory to scan.

    Returns:
        list[str]: Absolute paths to image files, sorted by filename.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names = sorted([n for n in os.listdir(folder) if n.lower().endswith(valid_exts)])
    return [os.path.join(folder, n) for n in names]


def preprocess_frame(frame_bgr, img_size):
    """Resize a BGR frame and convert it to a normalised float tensor.

    Args:
        frame_bgr (numpy.ndarray): Input frame in BGR format.
        img_size (int): Target width and height after resizing.

    Returns:
        torch.Tensor: RGB float tensor of shape (3, img_size, img_size)
        with values in [0, 1].
    """
    resized = cv2.resize(frame_bgr, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def colorize_mask(mask, color):
    """Create a 3-channel BGR image with active mask pixels set to a colour.

    Args:
        mask (numpy.ndarray): Binary mask of shape (H, W), dtype uint8.
        color (tuple[int, int, int]): BGR colour for active pixels.

    Returns:
        numpy.ndarray: BGR image of shape (H, W, 3).
    """
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask > 0] = color
    return canvas


def blend(frame, mask_color, alpha=0.45):
    """Alpha-blend a coloured mask onto a frame.

    Args:
        frame (numpy.ndarray): Background BGR frame.
        mask_color (numpy.ndarray): Coloured mask BGR image, same size as frame.
        alpha (float): Weight given to the mask layer (0 = invisible, 1 = opaque).

    Returns:
        numpy.ndarray: Blended BGR image.
    """
    return cv2.addWeighted(frame, 1.0 - alpha, mask_color, alpha, 0)


def parse_args():
    """Parse and return command-line arguments for test visualization.

    Returns:
        argparse.Namespace: Parsed arguments including model type, weights
        path, frame/mask directories, output directory, sequence settings,
        threshold, max samples, and blend alpha.
    """
    parser = argparse.ArgumentParser(
        description="Save side-by-side test visualizations: original, prediction, and ground truth."
    )
    parser.add_argument("--model_type", default="convlstm", choices=["convlstm", "cnn"])
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--frames_dir", default="dataset/frames/seq1", type=str)
    parser.add_argument("--masks_dir", default="dataset/masks/seq1", type=str)
    parser.add_argument("--output_dir", default="test_visualizations", type=str)
    parser.add_argument("--seq_len", default=5, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--threshold", default=0.35, type=float)
    parser.add_argument("--max_samples", default=0, type=int)
    parser.add_argument("--alpha", default=0.45, type=float)
    return parser.parse_args()


def main():
    """Run the test visualization pipeline.

    For each valid sliding-window sequence in the frames directory, runs
    model inference on the sequence and saves a side-by-side panel image
    containing the original frame, the predicted lane overlay, and the
    ground-truth mask from the masks directory.

    Raises:
        FileNotFoundError: If frames_dir, masks_dir, or the weights file
            cannot be found.
        ValueError: If fewer frames are available than seq_len.
    """
    args = parse_args()

    if not os.path.isdir(args.frames_dir):
        raise FileNotFoundError(f"frames_dir not found: {args.frames_dir}")
    if not os.path.isdir(args.masks_dir):
        raise FileNotFoundError(f"masks_dir not found: {args.masks_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = resolve_weights_path(args.model_type, args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Train the {args.model_type} model first or pass --weights explicitly."
        )

    model = get_model(args.model_type).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    frame_paths = sorted_image_paths(args.frames_dir)
    if len(frame_paths) < args.seq_len:
        raise ValueError(
            f"Need at least seq_len={args.seq_len} frames in {args.frames_dir}, found {len(frame_paths)}."
        )

    sample_count = 0

    with torch.no_grad():
        for i in range(len(frame_paths) - args.seq_len + 1):
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break

            seq_paths = frame_paths[i : i + args.seq_len]
            tensors = []

            for path in seq_paths:
                frame_bgr = cv2.imread(path)
                if frame_bgr is None:
                    tensors = []
                    break
                tensors.append(preprocess_frame(frame_bgr, args.img_size))

            if len(tensors) != args.seq_len:
                continue

            input_tensor = torch.stack(tensors).unsqueeze(0).to(device)
            pred = model(input_tensor)
            prob = pred[0, 0].cpu().numpy()

            frame_path = seq_paths[-1]
            frame_name = os.path.basename(frame_path)
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                continue

            h, w = frame_bgr.shape[:2]
            pred_mask = (prob >= args.threshold).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            mask_path = resolve_mask_path(args.masks_dir, frame_name)
            if mask_path is None:
                continue

            gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is None:
                continue
            gt_mask_raw = cv2.resize(gt_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_mask = gt_mask_raw.copy()
            gt_mask = (gt_mask >= 127).astype(np.uint8)

            pred_color = colorize_mask(pred_mask, (0, 0, 255))
            gt_color = colorize_mask(gt_mask, (255, 0, 0))

            pred_overlay = blend(frame_bgr, pred_color, alpha=args.alpha)
            # Show the actual ground-truth mask image loaded from masks_dir.
            gt_view = cv2.cvtColor(gt_mask_raw, cv2.COLOR_GRAY2BGR)

            # Add simple labels for easy visual inspection.
            cv2.putText(frame_bgr, "Original", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(pred_overlay, "Prediction", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(gt_view, "Ground Truth", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            panel = np.hstack([frame_bgr, pred_overlay, gt_view])

            out_name = f"sample_{sample_count:04d}_{frame_name}"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, panel)
            sample_count += 1

    print(f"Saved {sample_count} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
