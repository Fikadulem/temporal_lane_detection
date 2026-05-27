import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LaneVideoDataset(Dataset):

    def __init__(self, frames_dir, masks_dir, seq_len=5, img_size=(224, 224)):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.samples = []

        videos = os.listdir(frames_dir)

        for video in videos:

            frame_path = os.path.join(frames_dir, video)
            mask_path = os.path.join(masks_dir, video)

            frames = sorted(os.listdir(frame_path))

            # Include the equal-length case (e.g., 5 frames with seq_len=5).
            for i in range(len(frames) - seq_len + 1):
                self.samples.append(
                    (frame_path, mask_path, frames[i:i + seq_len])
                )

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def read_image(self, path):

        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img

    def read_mask(self, path):

        mask = cv2.imread(path, 0)
        mask = cv2.resize(mask, self.img_size)
        mask = torch.tensor(mask).float() / 255
        mask = mask.unsqueeze(0)

        return mask

    def __getitem__(self, idx):

        frame_dir, mask_dir, frame_names = self.samples[idx]

        frames = []
        masks = []

        for f in frame_names:

            frame_path = os.path.join(frame_dir, f)
            # Prefer same filename as the frame; fall back to common jpg/png swaps.
            mask_path = os.path.join(mask_dir, f)
            if not os.path.exists(mask_path):
                if f.lower().endswith(".jpg"):
                    mask_path = os.path.join(mask_dir, f[:-4] + ".png")
                elif f.lower().endswith(".png"):
                    mask_path = os.path.join(mask_dir, f[:-4] + ".jpg")

            frames.append(self.read_image(frame_path))
            masks.append(self.read_mask(mask_path))

        frames = torch.stack(frames)
        masks = torch.stack(masks)

        return frames, masks