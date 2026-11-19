
import os
from flask import Flask


def create_app():

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
    STATIC_DIR = os.path.join(BASE_DIR, "static")

    app = Flask(
        __name__,
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR
    )

    app.secret_key = "ppl_local_demo_key"

    app.config["BASE_DIR"] = BASE_DIR

    app.config["UPLOAD_FOLDER"] = os.path.join(
        BASE_DIR, "uploads"
    )

    app.config["RESULT_FOLDER"] = os.path.join(
        BASE_DIR, "results"
    )

    app.config["MASK_PATH"] = os.path.join(
        BASE_DIR,
        "models",
        "mask",
        "obstruction_mask_720p_refined.png"
    )

    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

    print("BASE DIRECTORY:", BASE_DIR)
    print("TEMPLATE DIRECTORY:", TEMPLATE_DIR)
    print("INDEX EXISTS:", os.path.exists(
        os.path.join(TEMPLATE_DIR, "index.html")
    ))
    print("MASK EXISTS:", os.path.exists(
        app.config["MASK_PATH"]
    ))

    from app.routes import main
    app.register_blueprint(main)

    return app
