"""Métricas por instância no estilo do artigo-base (Santos et al., 2020).

Casamento guloso um-para-um por ordem de confiança: cada predição casa com a anotação
ainda livre de maior IoU, se IoU >= limiar. Daí saem TP, FP, FN, precisão, revocação e F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_image_count_error: list[int] = field(default_factory=list)

    def add(self, tp: int, n_pred: int, n_gt: int) -> None:
        self.tp += tp
        self.fp += n_pred - tp
        self.fn += n_gt - tp
        self.per_image_count_error.append(n_pred - n_gt)

    def summary(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        mae = float(np.mean(np.abs(self.per_image_count_error))) if self.per_image_count_error else 0.0
        return {"TP": self.tp, "FP": self.fp, "FN": self.fn, "P": p, "R": r, "F1": f1, "MAE_contagem": mae}


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre caixas xyxy: a (N, 4), b (M, 4) -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def mask_iou_matrix(pred: list[np.ndarray], gt: list[np.ndarray]) -> np.ndarray:
    """IoU entre máscaras booleanas, calculado só onde as caixas envolventes se sobrepõem."""
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)))
    pb, gb = masks_to_boxes(pred), masks_to_boxes(gt)
    candidates = box_iou_matrix(pb, gb) > 0
    pred_area = [m.sum() for m in pred]
    gt_area = [m.sum() for m in gt]
    iou = np.zeros((len(pred), len(gt)))
    for i, j in zip(*np.nonzero(candidates)):
        x1, y1 = int(min(pb[i, 0], gb[j, 0])), int(min(pb[i, 1], gb[j, 1]))
        x2, y2 = int(max(pb[i, 2], gb[j, 2])) + 1, int(max(pb[i, 3], gb[j, 3])) + 1
        inter = np.logical_and(pred[i][y1:y2, x1:x2], gt[j][y1:y2, x1:x2]).sum()
        iou[i, j] = inter / (pred_area[i] + gt_area[j] - inter + 1e-9)
    return iou


def masks_to_boxes(masks: list[np.ndarray]) -> np.ndarray:
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [0, 0, 0, 0])
    return np.asarray(boxes, dtype=float)


def greedy_match(iou: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    """Número de verdadeiros positivos no casamento um-para-um."""
    if iou.size == 0:
        return 0
    taken = np.zeros(iou.shape[1], dtype=bool)
    tp = 0
    for i in np.argsort(-scores):
        candidates = np.where(~taken, iou[i], -1)
        j = int(np.argmax(candidates))
        if candidates[j] >= threshold:
            taken[j] = True
            tp += 1
    return tp
