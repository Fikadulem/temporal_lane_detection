import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LaneVideoDataset
from model import LaneDetectionModel


frames_path = "dataset/frames"
masks_path = "dataset/masks"

dataset = LaneVideoDataset(frames_path, masks_path, seq_len=5)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LaneDetectionModel().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10

for epoch in range(epochs):

    for frames, masks in loader:

        frames = frames.to(device)
        masks = masks[:, -1].to(device)

        preds = model(frames)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "lane_model.pth")