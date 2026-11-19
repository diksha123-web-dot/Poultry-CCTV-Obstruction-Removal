
import os
import time
import subprocess
import sys
import cv2


def run_propainter(video_path, mask_path, output_folder):

    start_time = time.time()

    project_folder = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    repo_folder = os.path.join(
        project_folder,
        "external",
        "ProPainter"
    )

    if not os.path.isfile(video_path):
        raise FileNotFoundError("Input CCTV video not found.")

    if not os.path.isfile(mask_path):
        raise FileNotFoundError("Obstruction mask not found.")

    if not os.path.isfile(
        os.path.join(repo_folder, "inference_propainter.py")
    ):
        raise FileNotFoundError("ProPainter repository not found.")

    os.makedirs(output_folder, exist_ok=True)

    command = [
        sys.executable,
        "inference_propainter.py",
        "-i", os.path.abspath(video_path),
        "-m", os.path.abspath(mask_path),
        "-o", os.path.abspath(output_folder),
        "--width", "432",
        "--height", "240",
        "--subvideo_length", "10",
        "--neighbor_length", "3",
        "--ref_stride", "10",
        "--mask_dilation", "2",
        "--fp16"
    ]

    print("Starting ProPainter processing...")

    process = subprocess.run(
        command,
        cwd=repo_folder,
        capture_output=True,
        text=True,
        timeout=1800
    )

    if process.returncode != 0:
        raise RuntimeError(
            "ProPainter failed:\n"
            + process.stderr[-3000:]
        )

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_video = os.path.join(
        output_folder,
        video_name,
        "inpaint_out.mp4"
    )

    if not os.path.isfile(output_video):
        raise FileNotFoundError(
            "ProPainter output video was not found: "
            + output_video
        )

    cap = cv2.VideoCapture(output_video)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open ProPainter output video."
        )

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    ret, frame = cap.read()
    cap.release()

    if frames <= 0 or not ret:
        raise RuntimeError(
            "ProPainter output contains no readable frames."
        )

    preview_path = os.path.join(
        output_folder,
        "propainter_output_preview.jpg"
    )

    if not cv2.imwrite(preview_path, frame):
        raise RuntimeError(
            "Could not save ProPainter preview image."
        )

    processing_time = time.time() - start_time

    average_fps = (
        frames / processing_time
        if processing_time > 0 else 0
    )

    print("ProPainter completed successfully!")
    print("Frames processed:", frames)
    print("Processing time:", round(processing_time, 2))
    print("Average FPS:", round(average_fps, 2))

    return {
        "video": output_video,
        "preview": preview_path,
        "frames": frames,
        "time": round(processing_time, 2),
        "fps": round(average_fps, 2)
    }
