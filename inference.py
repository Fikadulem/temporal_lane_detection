import cv2
import torch
import numpy as np
import os
from collections import deque

from model import LaneDetectionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LaneDetectionModel().to(device)
model.load_state_dict(torch.load("lane_model.pth", map_location=device))
model.eval()


def draw_classic_lanes(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)

    h, w = edges.shape
    roi = np.zeros_like(edges)
    polygon = np.array([
        [int(0.1 * w), h],
        [int(0.45 * w), int(0.6 * h)],
        [int(0.55 * w), int(0.6 * h)],
        [int(0.9 * w), h]
    ], dtype=np.int32)
    cv2.fillPoly(roi, [polygon], 255)
    masked_edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=100
    )

    overlay = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)

    return cv2.addWeighted(frame, 0.8, overlay, 0.4, 0)


def draw_detected_lane_lines(frame, prob, threshold=0.35):

    h, w = frame.shape[:2]
    mask = cv2.resize((prob > threshold).astype(np.uint8), (w, h))

    y_start = int(0.55 * h)
    ys, xs = np.where(mask[y_start:, :] > 0)

    if len(xs) < 50:
        return frame, 0

    ys = ys + y_start
    left_idx = xs < (w // 2)
    right_idx = xs >= (w // 2)

    overlay = frame.copy()
    lines_drawn = 0

    def fit_and_draw(x_points, y_points, color):
        if len(x_points) < 25:
            return 0

        m, b = np.polyfit(y_points, x_points, 1)
        y1 = h - 1
        y2 = int(0.6 * h)
        x1 = int(m * y1 + b)
        x2 = int(m * y2 + b)

        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))

        cv2.line(overlay, (x1, y1), (x2, y2), color, 6)
        return 1

    lines_drawn += fit_and_draw(xs[left_idx], ys[left_idx], (0, 255, 0))
    lines_drawn += fit_and_draw(xs[right_idx], ys[right_idx], (0, 255, 0))

    return cv2.addWeighted(frame, 0.75, overlay, 0.45, 0), lines_drawn


sequence = deque(maxlen=5)
output_dir = "inference_output"
os.makedirs(output_dir, exist_ok=True)
frame_idx = 0

cap = cv2.VideoCapture("challenge (1).mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.tensor(rgb).permute(2, 0, 1).float() / 255

    sequence.append(tensor)

    # Always extract visible lane lines using classical CV.
    frame_with_lanes = draw_classic_lanes(frame)

    if len(sequence) == 5:

        input_tensor = torch.stack(list(sequence))
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            pred = model(input_tensor)

        prob = pred[0, 0].cpu().numpy()
        lane_ratio = float((prob > 0.35).mean())

        if lane_ratio > 0.002:
            frame_with_lanes, detected_lines = draw_detected_lane_lines(
                frame_with_lanes,
                prob,
                threshold=0.35
            )

            if detected_lines == 0:
                mask = (prob > 0.35).astype(np.uint8) * 255
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                overlay = frame_with_lanes.copy()
                overlay[mask > 0] = [0, 255, 0]
                frame_with_lanes = cv2.addWeighted(frame_with_lanes, 0.7, overlay, 0.3, 0)

    frame = frame_with_lanes

    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
    cv2.imwrite(output_path, frame)
    frame_idx += 1

    cv2.imshow("Lane Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {frame_idx} frames to {output_dir}")