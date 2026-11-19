
import os
import glob

from services.lama_service import run_lama

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

video_files = []

for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
    video_files.extend(
        glob.glob(os.path.join(UPLOAD_FOLDER, ext))
    )

if not video_files:
    raise FileNotFoundError("No CCTV video found!")

VIDEO_PATH = max(
    video_files,
    key=os.path.getmtime
)

OUTPUT_PATH = os.path.join(
    RESULT_FOLDER,
    "lama_output.mp4"
)

print("Input:", VIDEO_PATH)

result = run_lama(
    VIDEO_PATH,
    MASK_PATH,
    OUTPUT_PATH
)

print("\nFINAL RESULT")
print(result)
