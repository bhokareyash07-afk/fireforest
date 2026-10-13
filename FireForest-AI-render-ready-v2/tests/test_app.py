from io import BytesIO

import app as app_module
from PIL import Image


def _accepted_report(_stream, _config):
    image = Image.new("RGB", (160, 140), (36, 138, 36))
    output = BytesIO(); image.save(output, "JPEG")
    return {"accepted": True, "status": "accepted", "message": "ok", "normalized_bytes": output.getvalue()}


def test_welcome_and_upload_pages_render(tmp_path):
    application = app_module.create_app({"TESTING": True, "UPLOAD_ROOT": tmp_path / "uploads", "OUTPUT_ROOT": tmp_path / "outputs", "RETENTION_HOURS": 0})
    client = application.test_client()
    assert client.get("/").status_code == 200
    assert b"Forest Fire Damage Assessment" in client.get("/").data
    assert client.get("/upload").status_code == 200


def test_preview_api_does_not_expose_normalised_image_bytes(tmp_path, monkeypatch):
    application = app_module.create_app({"TESTING": True, "UPLOAD_ROOT": tmp_path / "uploads", "OUTPUT_ROOT": tmp_path / "outputs", "RETENTION_HOURS": 0})
    monkeypatch.setattr(app_module, "validate_forest_image", lambda stream, config: {"accepted": True, "status": "accepted", "message": "ok", "normalized_bytes": b"private"})
    response = application.test_client().post("/api/validate-image", data={"image": (BytesIO(b"x"), "scene.jpg")})
    assert response.status_code == 200
    assert b"normalized_bytes" not in response.data


def test_complete_analysis_generates_results_and_download(tmp_path, monkeypatch):
    application = app_module.create_app({"TESTING": True, "UPLOAD_ROOT": tmp_path / "uploads", "OUTPUT_ROOT": tmp_path / "outputs", "RETENTION_HOURS": 0})
    monkeypatch.setattr(app_module, "validate_forest_image", _accepted_report)
    client = application.test_client()
    payload = {"before_image": (BytesIO(b"before"), "before.jpg"), "after_image": (BytesIO(b"after"), "after.jpg")}
    response = client.post("/analyze", data=payload, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Analysis statistics" in response.data
    analysis_id = next((tmp_path / "outputs").iterdir()).name
    download = client.get(f"/analysis/{analysis_id}/download")
    assert download.status_code == 200
    assert download.mimetype == "image/png"
    assert len(download.data) > 100
