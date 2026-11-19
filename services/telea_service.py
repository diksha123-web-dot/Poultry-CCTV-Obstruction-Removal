
import cv2
import os
import time
import numpy as np
import subprocess
import imageio_ffmpeg


def run_telea(video_path, mask_path, output_path):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError("Obstruction mask not found!")

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open input video!")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if mask.shape != (height, width):
        if abs(mask.shape[1] / mask.shape[0] - width / height) > 0.001:
            cap.release()
            raise ValueError("Mask and video aspect ratios do not match!")

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    # Temporary AVI file
    temp_video = os.path.splitext(output_path)[0] + "_temp.avi"

    writer = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create temporary video!")

    start_time = time.time()

    frames = 0
    first_result = None

    try:
        while frames < 250:

            ret, frame = cap.read()

            if not ret:
                break

            result = cv2.inpaint(
                frame,
                mask,
                5,
                cv2.INPAINT_TELEA
            )

            if first_result is None:
                first_result = result.copy()

            writer.write(result)
            frames += 1

    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raise ValueError("No frames processed!")

    # Convert AVI into browser-compatible H.264 MP4
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", temp_video,
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

    if conversion.returncode != 0:
        raise RuntimeError(
            "Video conversion failed:\n" + conversion.stderr[-2000:]
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError("Final MP4 video was not generated!")

    # Save preview image
    first_frame_path = (
        os.path.splitext(output_path)[0] + "_preview.jpg"
    )

    cv2.imwrite(first_frame_path, first_result)

    if os.path.exists(temp_video):
        os.remove(temp_video)

    processing_time = time.time() - start_time

    return {
        "frames": frames,
        "time": round(processing_time, 2),
        "fps": round(frames / processing_time, 2),
        "preview": first_frame_path,
        "video": output_path
    }
