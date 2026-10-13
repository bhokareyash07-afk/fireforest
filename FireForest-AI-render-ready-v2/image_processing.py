"""Pixel-level before/after comparison and heatmap generation.

This is deliberately a visual-change pipeline, not a calibrated burn-severity model.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ProcessingError(RuntimeError):
    pass


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ProcessingError("One of the saved images could not be read for analysis.")
    return image


def _try_affine_alignment(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """Attempt ECC alignment; return a clear warning if it is not reliable."""
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    # ECC is intentionally conservative: failed/low-correlation alignment is never silently claimed.
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 70, 1e-5)
    try:
        correlation, warp = cv2.findTransformECC(
            reference_gray.astype(np.float32) / 255.0,
            moving_gray.astype(np.float32) / 255.0,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        if correlation >= 0.42:
            aligned = cv2.warpAffine(
                moving,
                warp,
                (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned, True, f"Basic geometric alignment applied (ECC correlation {correlation:.2f})."
    except cv2.error:
        pass
    return moving, False, "Images were resized to a common frame, but reliable image registration was not achieved. Camera angle or alignment may affect detected changes."


def _difference_intensity(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_smooth = cv2.bilateralFilter(before, 7, 45, 45)
    after_smooth = cv2.bilateralFilter(after, 7, 45, 45)
    before_lab = cv2.cvtColor(before_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    after_lab = cv2.cvtColor(after_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Euclidean distance across L*a*b* carries luminance and colour change.
    difference = np.linalg.norm(before_lab - after_lab, axis=2)
    difference = np.clip(difference, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(difference, (5, 5), 0)


def _severity_masks(score: np.ndarray, change_threshold: float, moderate_threshold: float, high_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= change_threshold < moderate_threshold < high_threshold <= 255:
        raise ProcessingError("The configured change thresholds must be ordered between 0 and 255.")
    changed = score >= change_threshold
    blue = changed & (score < moderate_threshold)
    yellow = changed & (score >= moderate_threshold) & (score < high_threshold)
    red = changed & (score >= high_threshold)
    # Remove tiny components only from the common change mask, then reassign each retained pixel exactly once.
    clean = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return blue & clean, yellow & clean, red & clean


def _make_overlay(base: np.ndarray, blue: np.ndarray, yellow: np.ndarray, red: np.ndarray) -> np.ndarray:
    colour_layer = base.copy()
    # OpenCV is BGR. The labels/legend make the mapping transparent to users.
    colour_layer[blue] = (255, 0, 0)
    colour_layer[yellow] = (0, 255, 255)
    colour_layer[red] = (0, 0, 255)
    any_change = blue | yellow | red
    overlay = base.copy()
    blended = cv2.addWeighted(base, 0.36, colour_layer, 0.64, 0)
    overlay[any_change] = blended[any_change]
    return _append_legend(overlay)


def _append_legend(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    legend_height = 82
    canvas = np.full((height + legend_height, width, 3), (18, 41, 31), dtype=np.uint8)
    canvas[:height] = image
    labels = [("BLUE: lower", (255, 0, 0)), ("YELLOW: moderate", (0, 255, 255)), ("RED: highest", (0, 0, 255))]
    x = 18
    y = height + 21
    for text, colour in labels:
        cv2.rectangle(canvas, (x, y - 15), (x + 18, y + 3), colour, -1)
        cv2.putText(canvas, text, (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (244, 248, 244), 1, cv2.LINE_AA)
        y += 22
    return canvas


def run_damage_analysis(
    before_path: Path,
    after_path: Path,
    output_dir: Path,
    *,
    change_threshold: float,
    moderate_threshold: float,
    high_threshold: float,
) -> dict:
    before = _read_image(before_path)
    after = _read_image(after_path)
    original_before = {"width": int(before.shape[1]), "height": int(before.shape[0])}
    original_after = {"width": int(after.shape[1]), "height": int(after.shape[0])}
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
        resize_note = "The after-fire image was resized to match the before-fire image dimensions for pixel comparison."
    else:
        resize_note = "Both images already had matching dimensions."
    aligned_after, was_aligned, alignment_note = _try_affine_alignment(before, after)
    score = _difference_intensity(before, aligned_after)
    blue, yellow, red = _severity_masks(score, change_threshold, moderate_threshold, high_threshold)
    total_pixels = int(score.size)
    counts = {"blue": int(blue.sum()), "yellow": int(yellow.sum()), "red": int(red.sum())}
    changed_pixels = sum(counts.values())
    percentages = {
        "total": round(changed_pixels * 100 / total_pixels, 2),
        "red": round(counts["red"] * 100 / total_pixels, 2),
        "yellow": round(counts["yellow"] * 100 / total_pixels, 2),
        "blue": round(counts["blue"] * 100 / total_pixels, 2),
    }
    heatmap = _make_overlay(aligned_after, blue, yellow, red)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "damage_heatmap.png"
    if not cv2.imwrite(str(heatmap_path), heatmap):
        raise ProcessingError("The generated heatmap could not be saved.")
    return {
        "percentages": percentages,
        "pixel_counts": {**counts, "total_analyzed": total_pixels, "detected_change": changed_pixels},
        "dimensions": {"analyzed_width": int(score.shape[1]), "analyzed_height": int(score.shape[0]), "before_original": original_before, "after_original": original_after},
        "pipeline": {
            "resize_note": resize_note,
            "alignment_note": alignment_note,
            "alignment_applied": was_aligned,
            "change_threshold": change_threshold,
            "moderate_threshold": moderate_threshold,
            "high_threshold": high_threshold,
        },
    }
