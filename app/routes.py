
import os
import uuid
import cv2
import subprocess
import imageio_ffmpeg

from flask import (
    Blueprint, current_app, render_template,
    request, redirect, url_for, flash, send_from_directory
)

from werkzeug.utils import secure_filename

from services.telea_service import run_telea
from services.lama_service import run_lama
from services.propainter_service import run_propainter


main = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def create_10sec_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Cannot open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create 10-second video.")

    frames = 0
    max_frames = min(250, int(fps * 10))

    try:
        while frames < max_frames:

            ret, frame = cap.read()

            if not ret:
                break

            writer.write(frame)
            frames += 1

    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raise ValueError("No readable frames found.")

    return frames


def convert_h264(input_path):

    temp_path = input_path + ".converted.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        temp_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Video conversion failed: " + result.stderr[-2000:]
        )

    os.replace(temp_path, input_path)


@main.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        video = request.files.get("video")

        if not video or not video.filename:
            flash("Please select a CCTV video.")
            return redirect(url_for("main.index"))

        if not allowed_file(video.filename):
            flash("Unsupported video format.")
            return redirect(url_for("main.index"))

        mask_path = current_app.config["MASK_PATH"]

        if not os.path.isfile(mask_path):
            flash("Obstruction mask is missing.")
            return redirect(url_for("main.index"))

        job_id = uuid.uuid4().hex[:12]

        filename = secure_filename(video.filename)

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        result_folder = current_app.config["RESULT_FOLDER"]

        os.makedirs(upload_folder, exist_ok=True)

        uploaded_path = os.path.join(
            upload_folder,
            f"{job_id}_{filename}"
        )

        video.save(uploaded_path)

        job_folder = os.path.join(
            result_folder,
            job_id
        )

        os.makedirs(job_folder, exist_ok=True)

        original_path = os.path.join(
            job_folder,
            "original_10sec.mp4"
        )

        try:

            # STEP 1: Create common 10-second video

            frames = create_10sec_video(
                uploaded_path,
                original_path
            )

            convert_h264(original_path)

            print("Common input frames:", frames)


            # STEP 2: OpenCV Telea Inpainting

            print("Starting OpenCV Telea...")

            telea_path = os.path.join(
                job_folder,
                "telea_output.mp4"
            )

            telea = run_telea(
                original_path,
                mask_path,
                telea_path
            )

            print("Telea completed.")


            # STEP 3: ProPainter Video Inpainting

            print("Starting ProPainter...")

            propainter_folder = os.path.join(
                job_folder,
                "propainter"
            )

            propainter = run_propainter(
                original_path,
                mask_path,
                propainter_folder
            )

            print("ProPainter completed.")


            # STEP 4: LaMa Image Inpainting

            print("Starting LaMa...")

            lama_path = os.path.join(
                job_folder,
                "lama_output.mp4"
            )

            lama = run_lama(
                original_path,
                mask_path,
                lama_path
            )

            print("LaMa completed.")


            # STEP 5: Prepare ProPainter paths for HTML

            propainter_video = os.path.relpath(
                propainter["video"],
                job_folder
            ).replace("\\", "/")

            propainter_preview = os.path.relpath(
                propainter["preview"],
                job_folder
            ).replace("\\", "/")


            # STEP 6: GPU memory cleanup

            import gc
            import torch

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


            # STEP 7: Display comparison results

            return render_template(
                "results.html",

                job_id=job_id,

                telea=telea,
                lama=lama,
                propainter=propainter,

                propainter_video=propainter_video,
                propainter_preview=propainter_preview
            )


        except Exception as e:

            current_app.logger.exception(
                "CCTV processing failed"
            )

            flash(f"Processing failed: {e}")

            return redirect(url_for("main.index"))


    return render_template("index.html")


@main.route("/results/<job_id>/<path:filename>")
def result_file(job_id, filename):

    folder = os.path.join(
        current_app.config["RESULT_FOLDER"],
        job_id
    )

    return send_from_directory(folder, filename)
