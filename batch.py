"""
Batch-process a folder of images.

Usage:
  python batch.py <folder> [--threshold 0.3] [--out report.csv]
                          [--annotated-dir annotated/] [--recursive]
                          [--blur]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

from detector import Detector, annotate


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def iter_images(folder: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for p in sorted(folder.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch nudity detection on a folder")
    parser.add_argument("folder", type=Path, help="Folder of images to scan")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Minimum confidence to report (default 0.3)")
    parser.add_argument("--out", type=Path, default=Path("report.csv"),
                        help="CSV output path (default ./report.csv)")
    parser.add_argument("--annotated-dir", type=Path, default=None,
                        help="If set, save annotated images here")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories")
    parser.add_argument("--blur", action="store_true",
                        help="Blur explicit/suggestive regions in annotated output")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"error: not a directory: {args.folder}", file=sys.stderr)
        return 2

    if args.annotated_dir:
        args.annotated_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector()
    images = list(iter_images(args.folder, args.recursive))
    if not images:
        print(f"no images found in {args.folder}", file=sys.stderr)
        return 1

    print(f"scanning {len(images)} image(s)...")

    rows = []
    for i, path in enumerate(images, 1):
        try:
            result = detector.detect(path)
        except Exception as exc:
            print(f"  [{i}/{len(images)}] {path.name}: ERROR ({exc})")
            rows.append({
                "path": str(path), "verdict": "ERROR", "highest_score": "",
                "explicit": "", "detection_count": "", "labels": str(exc),
            })
            continue

        kept = result.filter(min_score=args.threshold)
        cats = {d.category for d in kept}
        if "explicit" in cats:
            verdict = "EXPLICIT"
        elif "suggestive" in cats:
            verdict = "SUGGESTIVE"
        elif kept:
            verdict = "NEUTRAL"
        else:
            verdict = "CLEAN"

        labels = ";".join(f"{d.class_name}:{d.score:.2f}" for d in kept)
        rows.append({
            "path": str(path),
            "verdict": verdict,
            "highest_score": f"{max((d.score for d in kept), default=0.0):.3f}",
            "explicit": "yes" if "explicit" in cats else "no",
            "detection_count": len(kept),
            "labels": labels,
        })
        print(f"  [{i}/{len(images)}] {path.name}: {verdict} ({len(kept)} hits)")

        if args.annotated_dir and kept:
            out_img = annotate(result.image_bgr, kept, blur=args.blur)
            out_path = args.annotated_dir / path.name
            cv2.imwrite(str(out_path), out_img)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "verdict", "highest_score", "explicit",
                        "detection_count", "labels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nreport written to {args.out}")
    if args.annotated_dir:
        print(f"annotated images written to {args.annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
