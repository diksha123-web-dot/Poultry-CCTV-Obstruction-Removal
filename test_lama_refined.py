
import os
import glob
import cv2
import numpy as np

from services.lama_service import test_lama_frame

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

# Find uploaded CCTV video
video_files = []

for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    video_files.extend(
        glob.glob(os.path.join(UPLOAD_FOLDER, ext))
    )

if not video_files:
    raise FileNotFoundError("No CCTV video found!")

VIDEO_PATH = max(video_files, key=os.path.getmtime)

# Read original mask
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError("Mask not found!")

# Slightly expand the white obstruction regions
kernel = np.ones((5, 5), np.uint8)

refined_mask = cv2.dilate(
    mask,
    kernel,
    iterations=1
)

# Save as a separate file
NEW_MASK_PATH = os.path.join(
    RESULT_FOLDER,
    "lama_refined_mask.png"
)

cv2.imwrite(NEW_MASK_PATH, refined_mask)

print("Refined mask saved:", NEW_MASK_PATH)

# Test LaMa on one frame
OUTPUT_PATH = os.path.join(
    RESULT_FOLDER,
    "lama_refined_preview.jpg"
)

test_lama_frame(
    VIDEO_PATH,
    NEW_MASK_PATH,
    OUTPUT_PATH
)

print("Refined LaMa test completed!")
print("Preview:", OUTPUT_PATH)
