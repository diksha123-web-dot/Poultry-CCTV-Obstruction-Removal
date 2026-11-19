
import os
import cv2
import time
import torch
import numpy as np

from PIL import Image
from simple_lama_inpainting import SimpleLama


# Load LaMa model only once
_lama_model = None


def get_lama_model():

    global _lama_model

    if _lama_model is None:

        print("Loading LaMa model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Using device:", device)

        _lama_model = SimpleLama(
            device=torch.device(device)
        )

        print("LaMa model loaded successfully!")

    return _lama_model


def test_lama_frame(video_path, mask_path, output_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open CCTV video!")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Unable to read first video frame!")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError("Obstruction mask not found!")

    height, width = frame.shape[:2]

    if abs(mask.shape[1] / mask.shape[0] - width / height) > 0.001:
        raise ValueError("Mask and video aspect ratios do not match!")

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask, (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(frame_rgb)
    mask_pil = Image.fromarray(mask, mode="L")

    lama = get_lama_model()
    result = lama(image_pil, mask_pil)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)

    print("LaMa test completed!")

    return output_path


def run_lama(video_path, mask_path, output_path):

    start_time = time.time()

    lama = get_lama_model()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Cannot open CCTV video!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    # Process maximum 250 frames (10 seconds at 25 FPS)
    total_frames = min(total_frames, 250)

    print("Video resolution:", width, "x", height)
    print("Frames to process:", total_frames)
    print("FPS:", fps)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        cap.release()
        raise ValueError("Cannot load obstruction mask!")

    if abs(mask.shape[1] / mask.shape[0] - width / height) > 0.001:
        cap.release()
        raise ValueError("Mask and video aspect ratios do not match!")

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    mask_pil = Image.fromarray(mask, mode="L")

    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    # Temporary AVI for reliable video writing
    temp_path = os.path.splitext(output_path)[0] + "_temp.avi"

    writer = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video!")

    frames_processed = 0

    preview_path = os.path.join(
        output_folder,
        "lama_output_preview.jpg"
    )

    try:

        while frames_processed < 250:

            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image_pil = Image.fromarray(frame_rgb)

            with torch.inference_mode():
                result = lama(image_pil, mask_pil)

            result_rgb = np.array(result.convert("RGB"))

            result_bgr = cv2.cvtColor(
                result_rgb,
                cv2.COLOR_RGB2BGR
            )

            if frames_processed == 0:
                cv2.imwrite(preview_path, result_bgr)

            writer.write(result_bgr)

            frames_processed += 1

            if frames_processed % 25 == 0:
                print(
                    "LaMa processed:",
                    frames_processed,
                    "/",
                    total_frames
                )

    finally:
        cap.release()
        writer.release()

    if frames_processed == 0:
        raise ValueError("No frames were processed by LaMa!")

    # Convert to browser-compatible H.264 MP4
    import imageio_ffmpeg
    import subprocess

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", temp_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path
    ]

    conversion = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if conversion.returncode != 0:
        raise RuntimeError(
            "LaMa video conversion failed:\n"
            + conversion.stderr[-2000:]
        )

    processing_time = time.time() - start_time

    average_fps = (
        frames_processed / processing_time
        if processing_time > 0 else 0
    )

    print("LaMa video processing completed!")
    print("Frames processed:", frames_processed)
    print("Processing time:", round(processing_time, 2))
    print("Average FPS:", round(average_fps, 2))

    return {
        "frames": frames_processed,
        "time": round(processing_time, 2),
        "fps": round(average_fps, 2),
        "video": output_path,
        "preview": preview_path
    }
