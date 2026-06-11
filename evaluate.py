import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LaneVideoDataset
from model import get_model


def resolve_weights_path(model_type, weights_path=None):
    """Resolve the checkpoint path to load.

    If an explicit path is provided it is used directly. Otherwise a default
    path is derived from the model type: ConvLSTM maps to ``lane_model.pth``
    and CNN to ``lane_model_<model_type>.pth``.

    Args:
        model_type (str): One of ``'convlstm'`` or ``'cnn'``.
        weights_path (str, optional): Explicit checkpoint path.

    Returns:
        str: Resolved path to the checkpoint file.
    """
    if weights_path is not None:
        return weights_path

    if model_type == "convlstm" and os.path.exists("lane_model.pth"):
        return "lane_model.pth"

    return f"lane_model_{model_type}.pth"


def compute_batch_metrics(preds, targets, threshold=0.35, eps=1e-8):
    """Compute pixel-level binary segmentation metrics for one batch.

    Binarises predictions at ``threshold`` and ground-truth at 0.5, then
    derives confusion-matrix counts and the derived metrics.

    Args:
        preds (torch.Tensor): Predicted probability map, shape (B, 1, H, W).
        targets (torch.Tensor): Ground-truth mask, shape (B, 1, H, W).
        threshold (float): Decision threshold applied to predictions.
        eps (float): Small constant to avoid division by zero.

    Returns:
        dict: Keys ``tp``, ``fp``, ``fn``, ``tn``, ``pixel_acc``,
            ``precision``, ``recall``, ``f1``, ``iou``, ``dice``.
    """
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= 0.5).float()

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1.0 - targets_bin)).sum()
    fn = ((1.0 - preds_bin) * targets_bin).sum()
    tn = ((1.0 - preds_bin) * (1.0 - targets_bin)).sum()

    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pixel_acc": pixel_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
    }


def evaluate(args):
    """Run full evaluation on the dataset and print results.

    Loads the model checkpoint, iterates over all sequence batches,
    accumulates pixel-level confusion counts across batches, and reports
    dataset-wide segmentation metrics (BCE loss, pixel accuracy, precision,
    recall, F1, IoU, Dice) together with inference speed statistics
    (latency per sample, sequences/s, frames/s).

    Args:
        args (argparse.Namespace): Parsed arguments from :func:`parse_args`.

    Raises:
        ValueError: If the dataset is empty or the checkpoint does not match
            the requested model type.
        FileNotFoundError: If the checkpoint file cannot be found.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LaneVideoDataset(
        args.frames_path,
        args.masks_path,
        seq_len=args.seq_len,
        img_size=(args.img_size, args.img_size),
    )

    if len(dataset) == 0:
        raise ValueError("Dataset has zero samples. Check paths and seq_len.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    weights_path = resolve_weights_path(args.model_type, args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Train the {args.model_type} model first or pass --weights explicitly."
        )

    model = get_model(args.model_type).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Provide a concise, actionable message for model/checkpoint mismatch.
        if args.model_type == "cnn" and any(k.startswith("convlstm.") for k in state_dict.keys()):
            raise ValueError(
                f"Checkpoint '{weights_path}' appears to be a ConvLSTM model, but --model_type is 'cnn'. "
                "Use --weights lane_model_cnn.pth or switch --model_type convlstm."
            ) from e
        if args.model_type == "convlstm" and any(k.startswith("temporal_projection.") for k in state_dict.keys()):
            raise ValueError(
                f"Checkpoint '{weights_path}' appears to be a CNN baseline model, but --model_type is 'convlstm'. "
                "Use --weights lane_model.pth (or lane_model_convlstm.pth) or switch --model_type cnn."
            ) from e
        raise ValueError(
            f"Failed to load checkpoint '{weights_path}' for model_type '{args.model_type}'. "
            "Verify the checkpoint was trained for the same model architecture."
        ) from e
    model.eval()

    total_tp = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)
    total_infer_time = 0.0
    total_samples = 0

    running_loss = 0.0
    num_batches = 0
    bce = torch.nn.BCELoss()

    with torch.no_grad():
        for frames, masks in loader:
            frames = frames.to(device)
            targets = masks[:, -1].to(device)

            start = time.perf_counter()
            preds = model(frames)
            total_infer_time += time.perf_counter() - start
            total_samples += frames.size(0)

            if preds.shape[-2:] != targets.shape[-2:]:
                preds = F.interpolate(
                    preds,
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            running_loss += bce(preds, targets).item()
            num_batches += 1

            m = compute_batch_metrics(preds, targets, threshold=args.threshold)
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            total_tn += m["tn"]

    eps = 1e-8
    pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    dice = 2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn + eps)
    avg_latency_ms = (total_infer_time / max(total_samples, 1)) * 1000.0
    seq_per_sec = total_samples / max(total_infer_time, eps)
    frames_per_sec = (total_samples * args.seq_len) / max(total_infer_time, eps)

    print("Evaluation Results")
    print("------------------")
    print(f"Samples: {len(dataset)}")
    print(f"Threshold: {args.threshold}")
    print(f"BCE Loss: {running_loss / max(num_batches, 1):.6f}")
    print(f"Pixel Accuracy: {pixel_acc.item():.6f}")
    print(f"Precision: {precision.item():.6f}")
    print(f"Recall: {recall.item():.6f}")
    print(f"F1 Score: {f1.item():.6f}")
    print(f"IoU: {iou.item():.6f}")
    print(f"Dice: {dice.item():.6f}")
    print("Inference Speed")
    print("---------------")
    print(f"Total Inference Time (s): {total_infer_time:.4f}")
    print(f"Avg Latency (ms/sample): {avg_latency_ms:.3f}")
    print(f"Throughput (seq/s): {seq_per_sec:.3f}")
    print(f"Throughput (frames/s): {frames_per_sec:.3f}")


def parse_args():
    """Parse and return command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments including dataset paths,
        checkpoint path, model type, sequence settings, and threshold.
    """
    parser = argparse.ArgumentParser(description="Evaluate lane segmentation against ground truth masks.")
    parser.add_argument("--frames_path", default="dataset/frames", type=str)
    parser.add_argument("--masks_path", default="dataset/masks", type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--seq_len", default=5, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--threshold", default=0.35, type=float)
    parser.add_argument("--model_type", default="convlstm", choices=["convlstm", "cnn"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
