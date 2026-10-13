from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from forest_validation import ImageValidationError, validate_forest_image


CONFIG = {"MIN_IMAGE_DIMENSION": 120, "MAX_IMAGE_DIMENSION": 6000}


def image_bytes(colour):
    image = Image.new("RGB", (180, 160), colour)
    output = BytesIO(); image.save(output, "JPEG")
    output.seek(0)
    return output


@patch("forest_validation._get_model_labels", return_value=[("valley", 0.91)])
def test_healthy_forest_style_image_can_be_accepted(_labels):
    report = validate_forest_image(image_bytes((35, 130, 38)), CONFIG)
    assert report["accepted"] is True


@patch("forest_validation._get_model_labels", return_value=[("volcano", 0.86)])
def test_burned_earth_style_image_can_be_accepted(_labels):
    report = validate_forest_image(image_bytes((105, 72, 45)), CONFIG)
    assert report["accepted"] is True


@patch("forest_validation._get_model_labels", return_value=[("sports car", 0.98)])
def test_unrelated_image_is_rejected(_labels):
    report = validate_forest_image(image_bytes((35, 150, 35)), CONFIG)
    assert report["accepted"] is False
    assert report["status"] == "invalid"


def test_corrupt_image_is_rejected():
    with pytest.raises(ImageValidationError):
        validate_forest_image(BytesIO(b"not an image"), CONFIG)
