"""Forest-scene validation using ImageNet MobileNetV2 plus conservative visual checks."""
from __future__ import annotations

import io
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


NEGATIVE_TERMS = {
    "car", "truck", "bus", "motor", "vehicle", "person", "face", "portrait", "suit", "gown",
    "plate", "pizza", "hotdog", "sandwich", "cheeseburger", "laptop", "monitor", "screen", "phone",
    "building", "palace", "church", "skyscraper", "castle", "shop", "book", "keyboard",
}
NATURE_TERMS = {
    "rainforest", "forest", "valley", "volcano", "cliff", "lakeside", "seashore", "promontory",
    "mountain", "alp", "meadow", "park", "wood", "coral_reef", "sandbar", "geyser",
}


@lru_cache(maxsize=1)
def _classifier():
    """Load once, on CPU. Keras obtains its small ImageNet weights on first use."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from tensorflow.keras.applications import MobileNetV2
        return MobileNetV2(weights="imagenet")
    except Exception as exc:
        raise ImageValidationError(
            "The image-validation model could not start. Install the project requirements and ensure the server can download the MobileNetV2 weights once."
        ) from exc


def _normalise_image(stream, config: dict) -> tuple[Image.Image, bytes]:
    try:
        stream.seek(0)
        raw = stream.read()
        if not raw:
            raise ImageValidationError("The uploaded file is empty.")
        with Image.open(io.BytesIO(raw)) as source:
            source.verify()
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("This file is not a readable image. Please choose a valid JPG, PNG, or WEBP image.") from exc
    width, height = image.size
    if min(width, height) < config["MIN_IMAGE_DIMENSION"]:
        raise ImageValidationError(f"Images must be at least {config['MIN_IMAGE_DIMENSION']} pixels wide and high.")
    if max(width, height) > config["MAX_IMAGE_DIMENSION"]:
        raise ImageValidationError(f"Images may not exceed {config['MAX_IMAGE_DIMENSION']} pixels on their longest side.")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, optimize=True)
    return image, output.getvalue()


def _get_model_labels(image: Image.Image) -> list[tuple[str, float]]:
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

    thumbnail = image.resize((224, 224))
    arr = preprocess_input(np.asarray(thumbnail, dtype=np.float32)[None, ...])
    predictions = _classifier().predict(arr, verbose=0)
    return [(label.replace("_", " ").lower(), float(score)) for _, label, score in decode_predictions(predictions, top=5)[0]]


def _visual_scene_metrics(image: Image.Image) -> dict[str, float]:
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # Green vegetation, dry/scorched earth, and textured natural terrain are all useful for the two stages.
    green = np.mean((hue >= 28) & (hue <= 95) & (saturation >= 38) & (value >= 35))
    earth = np.mean((hue >= 3) & (hue <= 35) & (saturation >= 22) & (value >= 20) & (value <= 220))
    dark = np.mean(value < 70)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    texture = min(cv2.Laplacian(grey, cv2.CV_64F).var() / 800.0, 1.0)
    score = min(1.0, 0.58 * green + 0.30 * earth + 0.12 * texture + 0.08 * dark)
    return {"green": float(green), "earth": float(earth), "dark": float(dark), "texture": float(texture), "natural_score": float(score)}


def _matches_any(label: str, terms: set[str]) -> bool:
    return any(term in label for term in terms)


def validate_forest_image(stream, config: dict) -> dict:
    """Return an explainable acceptance decision; never trust the name or extension alone."""
    image, normalised_bytes = _normalise_image(stream, config)
    labels = _get_model_labels(image)
    metrics = _visual_scene_metrics(image)
    negative = [(label, score) for label, score in labels if _matches_any(label, NEGATIVE_TERMS)]
    natural = [(label, score) for label, score in labels if _matches_any(label, NATURE_TERMS)]
    confidence = max((score for _, score in natural), default=0.0)
    # A strongly identified unrelated ImageNet category is rejected even if it has green/brown pixels.
    if negative and negative[0][1] >= config.get("FOREST_NEGATIVE_CONFIDENCE", 0.18):
        accepted, status = False, "invalid"
        message = "This appears to be an unrelated image rather than a forest scene. Please select a clear healthy-forest or fire-damaged forest image."
    elif (natural and confidence >= config.get("FOREST_MODEL_CONFIDENCE", 0.01) and (metrics["natural_score"] >= config.get("FOREST_MIN_NATURAL_SCORE", 0.12) or metrics["green"] >= 0.06)) or (metrics["green"] >= 0.16 and metrics["natural_score"] >= 0.17) or (metrics["earth"] >= 0.28 and metrics["texture"] >= 0.08 and not negative):
        accepted, status = True, "accepted"
        message = "Forest scene accepted. The image can be used for preliminary visual change analysis."
    else:
        accepted, status = False, "uncertain"
        message = "This image could not be confidently validated as a forest scene. Please use a clearer healthy-forest or fire-damaged forest photograph."
    return {
        "accepted": accepted,
        "status": status,
        "message": message,
        "model_labels": [{"label": label, "confidence": round(score * 100, 1)} for label, score in labels[:3]],
        "forest_model_confidence": round(confidence * 100, 1),
        "scene_metrics": {key: round(value * 100, 1) for key, value in metrics.items()},
        "normalized_bytes": normalised_bytes,
    }
