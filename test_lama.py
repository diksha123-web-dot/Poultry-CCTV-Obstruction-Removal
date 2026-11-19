
import os
import glob

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

OUTPUT_PATH = os.path.join(
    RESULT_FOLDER,
    "lama_test_preview.jpg"
)


# Find the most recently uploaded CCTV video
video_files = []

for extension in ("*.mp4", "*.avi", "*.mov", "*.mkv"):

    video_files.extend(
        glob.glob(os.path.join(UPLOAD_FOLDER, extension))
    )

if not video_files:
    raise FileNotFoundError(
        "No CCTV video found in uploads folder!"
    )

VIDEO_PATH = max(
    video_files,
    key=os.path.getmtime
)


print("Input video:", VIDEO_PATH)
print("Mask:", MASK_PATH)

test_lama_frame(
    VIDEO_PATH,
    MASK_PATH,
    OUTPUT_PATH
)
