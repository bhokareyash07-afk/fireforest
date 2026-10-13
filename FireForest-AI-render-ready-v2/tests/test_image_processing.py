from pathlib import Path

import cv2
import numpy as np

from image_processing import _make_overlay, _severity_masks, run_damage_analysis


def test_severity_masks_are_exclusive_and_complete_for_change_pixels():
    score = np.zeros((40, 160), dtype=np.uint8)
    score[5:35, 10:50] = 40
    score[5:35, 60:100] = 90
    score[5:35, 110:150] = 160
    blue, yellow, red = _severity_masks(score, 28, 72, 128)
    assert not np.any(blue & yellow)
    assert not np.any(blue & red)
    assert not np.any(yellow & red)
    assert np.all((blue | yellow | red)[score >= 28])
    assert blue.sum() > 0 and yellow.sum() > 0 and red.sum() > 0


def test_heatmap_overlay_contains_explicit_legend_colours():
    base = np.full((80, 220, 3), 80, dtype=np.uint8)
    blue = np.zeros((80, 220), dtype=bool); blue[10:25, 10:25] = True
    yellow = np.zeros((80, 220), dtype=bool); yellow[10:25, 40:55] = True
    red = np.zeros((80, 220), dtype=bool); red[10:25, 70:85] = True
    overlay = _make_overlay(base, blue, yellow, red)
    # The appended legend deliberately preserves the exact BGR category colours in the exported PNG.
    assert np.any(np.all(overlay == (255, 0, 0), axis=2))
    assert np.any(np.all(overlay == (0, 255, 255), axis=2))
    assert np.any(np.all(overlay == (0, 0, 255), axis=2))


def test_analysis_writes_heatmap_and_percentages_match_masks(tmp_path: Path):
    before = np.full((160, 220, 3), (35, 145, 35), dtype=np.uint8)
    after = before.copy()
    after[35:120, 50:170] = (35, 45, 125)  # a pronounced colour shift in a forest-like synthetic test image
    before_path, after_path = tmp_path / "before.jpg", tmp_path / "after.jpg"
    cv2.imwrite(str(before_path), before); cv2.imwrite(str(after_path), after)
    result = run_damage_analysis(before_path, after_path, tmp_path / "output", change_threshold=28, moderate_threshold=72, high_threshold=128)
    assert (tmp_path / "output" / "damage_heatmap.png").is_file()
    assert result["percentages"]["total"] == round(
        result["pixel_counts"]["detected_change"] * 100 / result["pixel_counts"]["total_analyzed"], 2
    )
    assert result["pixel_counts"]["detected_change"] == sum(result["pixel_counts"][key] for key in ("blue", "yellow", "red"))
