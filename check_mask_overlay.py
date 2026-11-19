
import os
import glob
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

MASK_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mask",
    "obstruction_mask_720p_refined.png"
)

RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_FOLDER, exist_ok=True)

# Find the latest uploaded video
video_files = []

for extension in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    video_files.extend(
        glob.glob(os.path.join(UPLOAD_FOLDER, extension))
    )

if not video_files:
    raise FileNotFoundError("No CCTV video found!")

VIDEO_PATH = max(video_files, key=os.path.getmtime)

# Read first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Cannot read CCTV video!")

# Read mask
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError("Obstruction mask not found!")

height, width = frame.shape[:2]

if mask.shape != (height, width):

    if abs(mask.shape[1] / mask.shape[0] - width / height) > 0.001:
        raise ValueError("Mask and video aspect ratios do not match!")

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

# Binary mask
mask = np.where(mask > 127, 255, 0).astype(np.uint8)

# Create red overlay on masked areas
overlay = frame.copy()

overlay[mask == 255] = [0, 0, 255]

# Blend original and overlay
result = cv2.addWeighted(
    frame, 0.65,
    overlay, 0.35,
    0
)

output_path = os.path.join(
    RESULT_FOLDER,
    "mask_overlay_preview.jpg"
)

cv2.imwrite(output_path, result)

print("Mask overlay created successfully!")
print("Output:", output_path)
print("Video resolution:", width, "x", height)
print(
    "Masked area:",
    round(np.count_nonzero(mask) / mask.size * 100, 2),
    "%"
)
