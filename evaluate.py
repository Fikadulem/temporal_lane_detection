import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LaneVideoDataset
from model import LaneDetectionModel


def compute_batch_metrics(preds, targets, threshold=0.35, eps=1e-8):
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

    model = LaneDetectionModel().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    total_tp = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)

    running_loss = 0.0
    num_batches = 0
    bce = torch.nn.BCELoss()

    with torch.no_grad():
        for frames, masks in loader:
            frames = frames.to(device)
            targets = masks[:, -1].to(device)

            preds = model(frames)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lane segmentation against ground truth masks.")
    parser.add_argument("--frames_path", default="dataset/frames", type=str)
    parser.add_argument("--masks_path", default="dataset/masks", type=str)
    parser.add_argument("--weights", default="lane_model.pth", type=str)
    parser.add_argument("--seq_len", default=5, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--threshold", default=0.35, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
