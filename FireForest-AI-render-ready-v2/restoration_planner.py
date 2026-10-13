"""Evidence-bound preliminary restoration guidance."""
from __future__ import annotations


def build_restoration_plan(percentages: dict[str, float]) -> list[dict[str, str]]:
    return [
        {
            "level": "Red regions",
            "coverage": f"{percentages['red']:.2f}% of analyzed image pixels",
            "advice": "Prioritize field inspection and a high-priority restoration assessment in visually highest-change locations.",
        },
        {
            "level": "Yellow regions",
            "coverage": f"{percentages['yellow']:.2f}% of analyzed image pixels",
            "advice": "Plan inspection, recovery monitoring, and ecological assessment before setting restoration actions.",
        },
        {
            "level": "Blue regions",
            "coverage": f"{percentages['blue']:.2f}% of analyzed image pixels",
            "advice": "Continue monitoring and compare against future photographs to establish whether visual change persists.",
        },
    ]
