"""Módulo geométrico: localiza bagas visíveis dentro da máscara de cada cacho e estima o diâmetro.

Versão revisada da função `encontrar_bagas_elite` do notebook.

Uso:
    python src/berries.py --weights weights/best.pt --source imagem.jpg --px-per-mm 4.2 --out resultados/bagas
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BerryParams:
    clahe_clip: float = 4.0
    min_dist: int = 15
    canny_high: int = 50
    accumulator: int = 25
    min_radius: int = 6
    max_radius: int = 32
    min_border_ratio: float = 0.5
    overlap_ratio: float = 0.8
    max_berries: int = 5


def find_berries(image_rgb: np.ndarray, cluster_mask: np.ndarray, p: BerryParams = BerryParams()) -> list[tuple[int, int, int]]:
    """Hough circular + distância à borda + supressão de sobreposição dentro de um cacho."""
    mask = cluster_mask.astype(np.uint8)
    gray = cv2.cvtColor(cv2.bitwise_and(image_rgb, image_rgb, mask=mask), cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.0, minDist=p.min_dist, param1=p.canny_high,
                               param2=p.accumulator, minRadius=p.min_radius, maxRadius=p.max_radius)
    if circles is None:
        return []

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidates = []
    for x, y, r in np.round(circles[0]).astype(int):
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] and distance[y, x] > r * p.min_border_ratio:
            candidates.append((x, y, r, distance[y, x]))
    candidates.sort(key=lambda c: c[3], reverse=True)

    kept: list[tuple[int, int, int]] = []
    for x1, y1, r1, _ in candidates:
        if all(np.hypot(x1 - x2, y1 - y2) >= (r1 + r2) * p.overlap_ratio for x2, y2, r2 in kept):
            kept.append((x1, y1, r1))
            if len(kept) >= p.max_berries:
                break
    return kept


def measure_image(model, image_path: Path, px_per_mm: float | None, conf: float = 0.25) -> list[dict]:
    result = model.predict(str(image_path), conf=conf, imgsz=640, retina_masks=True, verbose=False)[0]
    image_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
    rows = []
    if result.masks is None:
        return rows
    unit = "mm" if px_per_mm else "px"
    for i, mask in enumerate(result.masks.data.cpu().numpy()):
        berries = find_berries(image_rgb, mask > 0.5)
        diameters = [2 * r / px_per_mm if px_per_mm else 2 * r for _, _, r in berries]
        rows.append({
            "arquivo": image_path.name,
            "cacho_id": i + 1,
            "confianca": round(float(result.boxes.conf[i]), 3),
            "area_px": int((mask > 0.5).sum()),
            "n_bagas_medidas": len(diameters),
            f"diametro_medio_{unit}": round(float(np.mean(diameters)), 2) if diameters else None,
            f"diametro_min_{unit}": round(float(np.min(diameters)), 2) if diameters else None,
            f"diametro_max_{unit}": round(float(np.max(diameters)), 2) if diameters else None,
        })
    return rows


def main() -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", type=Path, required=True, help="Imagem ou pasta")
    parser.add_argument("--px-per-mm", type=float, help="Escala; sem ela o diâmetro sai em pixels")
    parser.add_argument("--out", type=Path, default=Path("resultados/bagas"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    images = sorted(args.source.glob("*.jpg")) if args.source.is_dir() else [args.source]
    rows = [row for image in images for row in measure_image(model, image, args.px_per_mm)]
    args.out.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({k for r in rows for k in r}, key=list(rows[0]).index)
        with open(args.out / "medidas_bagas.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} cachos medidos em {len(images)} imagem(ns) -> {args.out / 'medidas_bagas.csv'}")


if __name__ == "__main__":
    main()
