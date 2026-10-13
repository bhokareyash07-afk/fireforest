# FireForest AI

**Forest Fire Damage Assessment and Replantation Planning System** is a responsive Flask project for preliminary visual comparison of before-fire and after-fire forest photographs. It was built for the college project team:

- Mr. Suyash Kamble
- Mr. Arihant Choudhari
- Mr. Yash Bhokare
- Mr. Abhishek Patil

Project guide: **Prof. D. Patil**  
Institution: **Kasegaon Education Society's Rajarambapu Institute of Technology, Islampur**

> This is a visual-change tool, not a calibrated burn-severity or ecological-damage model. Field verification is required before restoration decisions.

## What is implemented

1. A responsive three-page flow: welcome page → upload and validation → separate persistent results page.
2. The requested college, team, guide, and quotation text on the welcome page.
3. Two image inputs, previews, removal/replacement, 10 MB limit, dimensions/type/content checks, loading feedback, and server-side revalidation.
4. CPU-only ImageNet **MobileNetV2** validation, combined conservatively with colour/texture scene signals. A confidently recognised unrelated category (for example, car, person, food, device, or building) is rejected; healthy green forests and brown/charred scenes can be accepted.
5. EXIF-aware image loading, size normalization, optional ECC geometric registration, bilateral/Gaussian noise reduction, L*a*b* pixel-difference calculation, morphology, mutually exclusive masks, and a generated PNG heatmap overlay.
6. Real image-area percentages using the total analyzed pixel count as denominator. Category masks are disjoint and every retained changed pixel has one category.
7. Dynamic result narrative, values, bar chart, restoration guidance, limitations, heatmap download, and a results URL that remains available until retention cleanup.
8. Safe generated filenames, per-analysis UUID folders, corrupt-image handling, no stack traces to users, and automatic 24-hour cleanup.
9. Automated tests for validation decisions, corrupt files, server response privacy, heatmap export/legend colours, percentage arithmetic, and page rendering.

## Project structure

```text
app.py                    Flask routes, upload handling, safe serving
forest_validation.py      MobileNetV2 + conservative scene validation
image_processing.py       Difference masks, percentages, heatmap PNG
restoration_planner.py    Evidence-bound restoration messages
templates/                Welcome, upload, results, and error pages
static/                   Responsive CSS and upload-page JavaScript
tests/                    Automated test suite
sample_data/              Sources and test checklist
uploads/                  Runtime uploads (not committed)
analysis_outputs/         Runtime results and heatmaps (not committed)
```

## Run locally

### Prerequisites

- Python 3.11 (recommended; Python 3.10–3.11 are best supported by the pinned TensorFlow CPU dependency)
- `pip`
- Internet access **once** on first valid upload so Keras can obtain MobileNetV2 ImageNet weights. Afterward Keras caches them in the host's model cache.

### Install and start

```bash
git clone <your-repository-url>
cd forest-fire-project
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env    # Windows; use cp on macOS/Linux
python app.py
```

Open `http://127.0.0.1:5000`. For development-only hot reloading, set `FLASK_DEBUG=1` in `.env` before starting. Do not use debug mode in deployment.

`college-logo.png` is deliberately not supplied. To show the official logo, obtain approval and put an authorized file at `static/images/college-logo.png`. The welcome page has a reserved logo area and falls back to a text marker when the file is absent.

## Configure environment variables

Copy `.env.example` and set these values in a local shell or hosting dashboard:

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Flask session signing key; use a long random value in production | development placeholder |
| `PORT` | Listening port | `5000` |
| `MAX_UPLOAD_BYTES` | Maximum request/image size | `10485760` (10 MB) |
| `MIN_IMAGE_DIMENSION` / `MAX_IMAGE_DIMENSION` | Image dimension guardrails | `120` / `6000` |
| `CHANGE_THRESHOLD` | Minimum L*a*b* visual difference considered changed | `28` |
| `MODERATE_THRESHOLD` / `HIGH_THRESHOLD` | Mask category boundaries, must be ordered | `72` / `128` |
| `FOREST_MODEL_CONFIDENCE` | Minimum ImageNet natural-scene confidence used with visual checks | `0.01` |
| `FOREST_NEGATIVE_CONFIDENCE` | Confidence that rejects unrelated ImageNet categories | `0.18` |
| `FOREST_MIN_NATURAL_SCORE` | Minimum combined vegetation/earth/texture signal | `0.12` |
| `RETENTION_HOURS` | Age at which UUID analysis folders are removed on app startup | `24` |

The numeric validation thresholds are deliberately configurable: different regions, seasons, and image sources need calibration. Keep the three visual-difference thresholds ordered: `CHANGE_THRESHOLD < MODERATE_THRESHOLD < HIGH_THRESHOLD`.

## How forest validation works—and its limits

`forest_validation.py` loads Keras Applications **MobileNetV2** with ImageNet weights on the CPU. It examines the top predicted labels alongside vegetation, dry/earth-tone, darkness, and texture measurements. A high-confidence unrelated label is rejected. Forest/natural-scene evidence plus conservative visual checks can accept healthy forests; charred scenes are not rejected merely for lacking green pixels.

This model was not trained specifically to label every forest-fire photograph. ImageNet labels may be uncertain for aerial imagery, unusual camera angles, smoke, nighttime scenes, or local ecosystem types. The app therefore shows an uncertainty message rather than silently accepting every file. It does **not** claim perfect accuracy or independently prove that vegetation is burned.

The heatmap is a preliminary *visual-change* overlay. Lighting, shadows, season, crop, camera position, cloud/smoke, and imperfect alignment can cause changes. Simple resizing is never described as registration; the app reports whether its separate ECC geometric-alignment attempt was reliable.

## Test it

```bash
pytest -q
```

The unit suite mocks MobileNet predictions where necessary to test known healthy, charred, and unrelated cases without requiring a model download. The `sample_data/README.md` file gives sources for manual image testing. Also manually verify the complete browser workflow with healthy/burned forest images; unrelated image pairs and a forest plus unrelated image; corrupt data; unsupported files; missing inputs; over-limit files; differing dimensions; heatmap download; repeat analyses; and mobile/desktop widths.

## Deploy to Render (public URL)

This repository contains a `Procfile`, `runtime.txt`, and `.python-version` suitable for Render's Python web service. The `.python-version` file pins Python 3.11.9; keep it at the repository root because TensorFlow CPU in this project is not intended for Render's newer default Python 3.14 runtime.

1. Put this project in a GitHub/GitLab repository. Do **not** commit `.env`, `uploads/`, or `analysis_outputs/`.
2. In Render, select **New → Web Service**, connect that repository, and choose the `main` branch.
3. Select Python 3.11. Use build command `pip install -r requirements.txt` and start command `gunicorn --workers 1 --threads 4 --timeout 120 'app:create_app()'`.
4. Set a production `SECRET_KEY`, keep `FLASK_DEBUG=0`, and set optional `.env.example` values as required.
5. Deploy. On first image validation, the host needs outbound access to download/cache MobileNetV2 weights. Pre-bundle/cache the Keras model if runtime downloads are blocked.
6. Test the deployed domain with the browser checklist before sharing its URL.

### Hosting notes

TensorFlow CPU and OpenCV need much more RAM than a static site; a small hobby instance may cold-start slowly or run out of memory. Limit the service to one worker unless it has enough RAM. Many free hosts use ephemeral storage, so uploads, heatmaps, and cached model weights may disappear after a restart. This fits the app's 24-hour temporary-file policy, but persistent history requires a database/object store.

## Future enhancements

- A forest/fire-damage classifier trained and evaluated on geographically representative labelled imagery.
- Satellite indices such as NBR/dNBR, georeferencing, and calibrated severity thresholds.
- Authentication, an analysis-history database, object storage, retention controls, and audit trails.
- Better multi-image feature matching, map view, field-data collection, and locally appropriate species recommendations reviewed by ecologists.
- Background task queues for large images and pre-bundled model weights for constrained hosting.

## Attribution

The welcome visual uses a remotely served forest photograph by [Casey Horner on Unsplash](https://unsplash.com/photos/4rDCa5hBlCs), under the [Unsplash License](https://unsplash.com/license). Its link and college-logo replacement guidance also appear in `static/images/README.md`.
