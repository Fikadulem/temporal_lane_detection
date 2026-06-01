import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LaneVideoDataset
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detection models.")
    parser.add_argument("--frames_path", default="dataset/frames", type=str)
    parser.add_argument("--masks_path", default="dataset/masks", type=str)
    parser.add_argument("--seq_len", default=5, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model_type", default="convlstm", choices=["convlstm", "cnn"])
    parser.add_argument("--weights_out", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = LaneVideoDataset(
        args.frames_path,
        args.masks_path,
        seq_len=args.seq_len,
        img_size=(args.img_size, args.img_size),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_type).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        for frames, masks in loader:

            frames = frames.to(device)
            masks = masks[:, -1].to(device)

            preds = model(frames)

            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch:", epoch, "Loss:", loss.item())

    if args.weights_out is not None:
        weights_out = args.weights_out
    elif args.model_type == "convlstm":
        weights_out = "lane_model.pth"
    else:
        weights_out = f"lane_model_{args.model_type}.pth"

    torch.save(model.state_dict(), weights_out)
    print("Saved weights to", weights_out)


if __name__ == "__main__":
    main()