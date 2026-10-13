"""FireForest AI Flask application."""
from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge

from forest_validation import ImageValidationError, validate_forest_image
from image_processing import ProcessingError, run_damage_analysis
from restoration_planner import build_restoration_plan
from utils import allowed_filename, cleanup_expired_directories, read_json, write_json


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
UPLOAD_ROOT = BASE_DIR / "uploads"
OUTPUT_ROOT = BASE_DIR / "analysis_outputs"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "change-this-development-key"),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)),
        MAX_IMAGE_DIMENSION=int(os.environ.get("MAX_IMAGE_DIMENSION", 6000)),
        MIN_IMAGE_DIMENSION=int(os.environ.get("MIN_IMAGE_DIMENSION", 120)),
        CHANGE_THRESHOLD=float(os.environ.get("CHANGE_THRESHOLD", 28)),
        MODERATE_THRESHOLD=float(os.environ.get("MODERATE_THRESHOLD", 72)),
        HIGH_THRESHOLD=float(os.environ.get("HIGH_THRESHOLD", 128)),
        FOREST_MODEL_CONFIDENCE=float(os.environ.get("FOREST_MODEL_CONFIDENCE", 0.01)),
        FOREST_NEGATIVE_CONFIDENCE=float(os.environ.get("FOREST_NEGATIVE_CONFIDENCE", 0.18)),
        FOREST_MIN_NATURAL_SCORE=float(os.environ.get("FOREST_MIN_NATURAL_SCORE", 0.12)),
        RETENTION_HOURS=int(os.environ.get("RETENTION_HOURS", 24)),
        UPLOAD_ROOT=UPLOAD_ROOT,
        OUTPUT_ROOT=OUTPUT_ROOT,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OUTPUT_ROOT"]).mkdir(parents=True, exist_ok=True)
    cleanup_expired_directories([Path(app.config["UPLOAD_ROOT"]), Path(app.config["OUTPUT_ROOT"])], app.config["RETENTION_HOURS"])

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/upload")
    def upload():
        return render_template("upload.html", max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))

    @app.post("/api/validate-image")
    def validate_image_api():
        image = request.files.get("image")
        if image is None or not image.filename:
            return jsonify(ok=False, status="invalid", message="Select an image first."), 400
        if not allowed_filename(image.filename, ALLOWED_EXTENSIONS):
            return jsonify(ok=False, status="invalid", message="Use JPG, JPEG, PNG, or WEBP images only."), 400
        try:
            report = validate_forest_image(image.stream, app.config)
            report.pop("normalized_bytes", None)
            return jsonify(ok=report["accepted"], **report), (200 if report["accepted"] else 422)
        except ImageValidationError as exc:
            return jsonify(ok=False, status="invalid", message=str(exc)), 422
        except Exception:
            app.logger.exception("Preview validation failed")
            return jsonify(ok=False, status="error", message="We could not validate that image. Please try a different clear image."), 500

    @app.post("/analyze")
    def analyze():
        before = request.files.get("before_image")
        after = request.files.get("after_image")
        if before is None or after is None or not before.filename or not after.filename:
            flash("Please select both a before-fire and an after-fire forest image.", "error")
            return redirect(url_for("upload"))
        if not allowed_filename(before.filename, ALLOWED_EXTENSIONS) or not allowed_filename(after.filename, ALLOWED_EXTENSIONS):
            flash("Only JPG, JPEG, PNG, and WEBP images are supported.", "error")
            return redirect(url_for("upload"))

        analysis_id = uuid.uuid4().hex
        upload_dir = Path(app.config["UPLOAD_ROOT"]) / analysis_id
        output_dir = Path(app.config["OUTPUT_ROOT"]) / analysis_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            before_report = validate_forest_image(before.stream, app.config)
            after_report = validate_forest_image(after.stream, app.config)
            if not before_report["accepted"] or not after_report["accepted"]:
                flash(
                    "Invalid Images: Forest Images Required. One or both uploaded images do not appear to contain a suitable forest scene. Please upload valid before-fire and after-fire images.",
                    "error",
                )
                return redirect(url_for("upload"))

            # Streams have been consumed by validation; save the normalized bytes returned by it.
            # Validation normalizes every accepted upload to JPEG, so keep the stored extension truthful.
            before_name = "before_image.jpg"
            after_name = "after_image.jpg"
            before_path = upload_dir / before_name
            after_path = upload_dir / after_name
            before_path.write_bytes(before_report.pop("normalized_bytes"))
            after_path.write_bytes(after_report.pop("normalized_bytes"))

            result = run_damage_analysis(
                before_path, after_path, output_dir,
                change_threshold=app.config["CHANGE_THRESHOLD"],
                moderate_threshold=app.config["MODERATE_THRESHOLD"],
                high_threshold=app.config["HIGH_THRESHOLD"],
            )
            result["analysis_id"] = analysis_id
            result["created_at"] = datetime.now(timezone.utc).isoformat()
            result["validation"] = {"before": before_report, "after": after_report}
            result["files"] = {"before": before_name, "after": after_name, "heatmap": "damage_heatmap.png"}
            result["restoration"] = build_restoration_plan(result["percentages"])
            write_json(output_dir / "analysis.json", result)
            session["latest_analysis_id"] = analysis_id
            return redirect(url_for("results", analysis_id=analysis_id))
        except (ImageValidationError, ProcessingError) as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            flash(str(exc), "error")
            return redirect(url_for("upload"))
        except Exception:
            app.logger.exception("Analysis failed")
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)
            flash("Analysis could not be completed. Please use two clear forest images and try again.", "error")
            return redirect(url_for("upload"))

    @app.get("/results/<analysis_id>")
    def results(analysis_id: str):
        data = _load_analysis(app, analysis_id)
        return render_template("results.html", analysis=data)

    @app.get("/analysis/<analysis_id>/image/<image_kind>")
    def analysis_image(analysis_id: str, image_kind: str):
        data = _load_analysis(app, analysis_id)
        if image_kind not in {"before", "after", "heatmap"}:
            abort(404)
        root = Path(app.config["OUTPUT_ROOT"]) if image_kind == "heatmap" else Path(app.config["UPLOAD_ROOT"])
        filename = data["files"][image_kind]
        directory = root / analysis_id
        file_path = directory / filename
        if not file_path.is_file():
            abort(404)
        return send_file(file_path, conditional=True, download_name=filename)

    @app.get("/analysis/<analysis_id>/download")
    def download_heatmap(analysis_id: str):
        _load_analysis(app, analysis_id)
        heatmap_path = Path(app.config["OUTPUT_ROOT"]) / analysis_id / "damage_heatmap.png"
        if not heatmap_path.is_file():
            abort(404)
        return send_file(heatmap_path, as_attachment=True, download_name=f"FireForest-AI-heatmap-{analysis_id[:8]}.png")

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        message = f"Image files must be smaller than {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."
        if request.path.startswith("/api/"):
            return jsonify(ok=False, status="invalid", message=message), 413
        flash(message, "error")
        return redirect(url_for("upload"))

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", title="Page not found", message="That analysis or page is no longer available."), 404

    return app


def _load_analysis(app: Flask, analysis_id: str) -> dict:
    if not analysis_id.isalnum() or len(analysis_id) != 32:
        abort(404)
    metadata = Path(app.config["OUTPUT_ROOT"]) / analysis_id / "analysis.json"
    if not metadata.is_file():
        abort(404)
    return read_json(metadata)


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
