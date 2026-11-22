"""Avalia baselines e modelo ajustado nas MESMAS condições de teste.

Métodos:
    b0  YOLOv8n-seg pré-treinado no COCO, sem ajuste (classe-agnóstico: qualquer detecção = cacho)
    b1  Regra HSV + morfologia (parâmetros escolhidos na validação com --tune-b1)
    ft  YOLOv8n-seg após fine-tuning no WGISD (--weights)

Conjuntos de teste (anotações do WGISD):
    test_masked  27 imagens, 408 cachos com máscara -> F1 de máscara e de caixa
    test         58 imagens, 850 cachos com caixa  -> F1 de caixa

Uso:
    python src/evaluate.py --wgisd ../wgisd --tune-b1
    python src/evaluate.py --wgisd ../wgisd --methods b0 b1 ft --weights weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from baseline_hsv import HSVRule, search_grid, segment_clusters
from metrics import Counts, box_iou_matrix, greedy_match, mask_iou_matrix, masks_to_boxes

IOU_THRESHOLD = 0.5
CONF = 0.25
IMGSZ = 640


def read_names(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]


def load_gt_boxes(data_dir: Path, name: str, width: int, height: int) -> np.ndarray:
    rows = np.loadtxt(data_dir / f"{name}.txt", ndmin=2)
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


@lru_cache(maxsize=32)
def load_gt_masks(data_dir: Path, name: str) -> list[np.ndarray]:
    masks = np.load(data_dir / f"{name}.npz")["arr_0"]
    return [masks[:, :, i] > 0 for i in range(masks.shape[2])]


class YoloPredictor:
    def __init__(self, weights: str):
        from ultralytics import YOLO

        self.model = YOLO(weights)

    def __call__(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        result = self.model.predict(image, imgsz=IMGSZ, conf=CONF, retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return [], np.zeros((0, 4)), np.zeros(0)
        masks = [m.astype(bool) for m in result.masks.data.cpu().numpy()]
        return masks, result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


class HSVPredictor:
    def __init__(self, rule: HSVRule):
        self.rule = rule

    def __call__(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        masks, scores = segment_clusters(image, self.rule)
        boxes = masks_to_boxes(masks) if masks else np.zeros((0, 4))
        return masks, boxes, scores


def evaluate(predictor, data_dir: Path, names: list[str], with_masks: bool) -> dict:
    box_counts, mask_counts = Counts(), Counts()
    start = time.perf_counter()
    for name in names:
        image = cv2.imread(str(data_dir / f"{name}.jpg"))
        height, width = image.shape[:2]
        pred_masks, pred_boxes, scores = predictor(image)
        gt_boxes = load_gt_boxes(data_dir, name, width, height)
        box_counts.add(greedy_match(box_iou_matrix(pred_boxes, gt_boxes), scores, IOU_THRESHOLD), len(scores), len(gt_boxes))
        if with_masks:
            gt_masks = load_gt_masks(data_dir, name)
            tp = greedy_match(mask_iou_matrix(pred_masks, gt_masks), scores, IOU_THRESHOLD)
            mask_counts.add(tp, len(scores), len(gt_masks))
    elapsed = time.perf_counter() - start
    out = {"imagens": len(names), "seg_por_imagem": elapsed / max(1, len(names)), "caixa": box_counts.summary()}
    if with_masks:
        out["mascara"] = mask_counts.summary()
    return out


def tune_b1(data_dir: Path, val_names: list[str], out_path: Path) -> HSVRule:
    best_rule, best_f1, log = None, -1.0, []
    for rule in search_grid():
        f1 = evaluate(HSVPredictor(rule), data_dir, val_names, with_masks=True)["mascara"]["F1"]
        log.append({**rule.as_dict(), "F1_mascara_val": f1})
        if f1 > best_f1:
            best_rule, best_f1 = rule, f1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"melhor": best_rule.as_dict(), "F1_val": best_f1, "grade": log}, indent=2))
    print(f"B1: melhor F1 de máscara na validação = {best_f1:.3f} com {best_rule}")
    return best_rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wgisd", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["b0", "b1", "ft"], choices=["b0", "b1", "ft"])
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--b1-params", type=Path, default=Path("resultados/metricas/b1_hsv_params.json"))
    parser.add_argument("--tune-b1", action="store_true")
    parser.add_argument("--val-list", type=Path, default=Path("data/b1_ajuste.txt"))
    parser.add_argument("--out", type=Path, default=Path("resultados/metricas"))
    args = parser.parse_args()

    data_dir = args.wgisd / "data"
    splits = {
        "test_masked": (read_names(args.wgisd / "test_masked.txt"), True),
        "test": (read_names(args.wgisd / "test.txt"), False),
    }

    predictors = {}
    if "b0" in args.methods:
        predictors["b0"] = YoloPredictor("yolov8n-seg.pt")
    if "b1" in args.methods:
        if args.tune_b1 or not args.b1_params.exists():
            rule = tune_b1(data_dir, read_names(args.val_list), args.b1_params)
        else:
            rule = HSVRule(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(args.b1_params.read_text())["melhor"].items()})
        predictors["b1"] = HSVPredictor(rule)
    if "ft" in args.methods:
        predictors["ft"] = YoloPredictor(args.weights)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, predictor in predictors.items():
        for split, (names, with_masks) in splits.items():
            result = evaluate(predictor, data_dir, names, with_masks)
            (args.out / f"{method}_{split}.json").write_text(json.dumps(result, indent=2))
            for kind in ("mascara", "caixa"):
                if kind in result:
                    rows.append({"metodo": method, "conjunto": split, "tipo": kind, **result[kind],
                                 "seg_por_imagem": round(result["seg_por_imagem"], 3)})
            print(method, split, json.dumps({k: result[k] for k in result if k in ("mascara", "caixa")}))

    with open(args.out / "comparacao.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
