"""Baseline B1: regra clássica sem aprendizado (HSV + morfologia + componentes conexos).

Os parâmetros são escolhidos por busca em grade na partição de VALIDAÇÃO .
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVRule:
    green_h: tuple[int, int] = (25, 45)
    green_s_min: int = 60
    dark_v_max: int = 70
    dark_s_min: int = 30
    kernel: int = 9
    min_area_frac: float = 0.0005

    def as_dict(self) -> dict:
        return asdict(self)


def segment_clusters(image_bgr: np.ndarray, rule: HSVRule) -> tuple[list[np.ndarray], np.ndarray]:
    """Retorna máscaras de instância e um escore (área relativa) para cada componente."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white_grape = (h >= rule.green_h[0]) & (h <= rule.green_h[1]) & (s >= rule.green_s_min) & (v > rule.dark_v_max)
    dark_grape = (v <= rule.dark_v_max) & (s >= rule.dark_s_min)
    mask = (white_grape | dark_grape).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rule.kernel, rule.kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = rule.min_area_frac * mask.size
    instances, scores = [], []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area >= min_area:
            instances.append(labels == k)
            scores.append(area / mask.size)
    return instances, np.asarray(scores, dtype=float)


def search_grid() -> list[HSVRule]:
    grid = itertools.product(
        [(25, 45), (20, 50)],
        [40, 80],
        [60, 80],
        [7, 13],
        [0.0005, 0.002],
    )
    return [
        HSVRule(green_h=gh, green_s_min=gs, dark_v_max=dv, kernel=k, min_area_frac=a)
        for gh, gs, dv, k, a in grid
    ]
