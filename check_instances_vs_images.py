#!/usr/bin/env python
"""Check COCO instances_{train,val}.json against image folders.

Designed for very large COCO JSONs (hundreds of MB): uses ijson to stream-parse
`images[].file_name` without loading full JSON into memory.

Usage:
  python check_instances_vs_images.py \
    --train-json instances_train.json --val-json instances_val.json \
    --train-images UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/train \
    --val-images   UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/val
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import ijson


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_coco_image_filenames(coco_json_path: Path) -> Iterator[str]:
    """Yield `images[].file_name` from a COCO json, streaming."""
    with coco_json_path.open("rb") as f:
        # This yields dicts for each element under images[]
        for image_obj in ijson.items(f, "images.item"):
            fn = image_obj.get("file_name")
            if isinstance(fn, str) and fn:
                yield fn


def list_image_files(images_dir: Path) -> list[str]:
    """List image file basenames under images_dir (recursive)."""
    files: list[str] = []
    for root, _, filenames in os.walk(images_dir):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                files.append(name)
    return files


def summarize_split(
    split_name: str,
    coco_json_path: Path,
    images_dir: Path,
) -> dict:
    json_names = list(iter_coco_image_filenames(coco_json_path))
    json_counts = Counter(json_names)
    json_set = set(json_counts.keys())

    disk_names = list_image_files(images_dir)
    disk_counts = Counter(disk_names)
    disk_set = set(disk_counts.keys())

    missing_on_disk = sorted(json_set - disk_set)
    extra_on_disk = sorted(disk_set - json_set)

    dup_in_json = {k: v for k, v in json_counts.items() if v > 1}
    dup_on_disk = {k: v for k, v in disk_counts.items() if v > 1}

    return {
        "split": split_name,
        "json_total": len(json_names),
        "json_unique": len(json_set),
        "disk_total": len(disk_names),
        "disk_unique": len(disk_set),
        "missing_on_disk": missing_on_disk,
        "extra_on_disk": extra_on_disk,
        "dup_in_json": dup_in_json,
        "dup_on_disk": dup_on_disk,
    }


def print_report(title: str, items: Iterable[str], max_items: int = 25) -> None:
    items = list(items)
    print(f"{title}: {len(items)}")
    for s in items[:max_items]:
        print(f"  - {s}")
    if len(items) > max_items:
        print(f"  ... ({len(items) - max_items} more)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", type=Path, required=True)
    ap.add_argument("--val-json", type=Path, required=True)
    ap.add_argument("--train-images", type=Path, required=True)
    ap.add_argument("--val-images", type=Path, required=True)
    ap.add_argument("--max-list", type=int, default=25)
    args = ap.parse_args()

    for p in [args.train_json, args.val_json, args.train_images, args.val_images]:
        if not p.exists():
            raise SystemExit(f"Path does not exist: {p}")

    train = summarize_split("train", args.train_json, args.train_images)
    val = summarize_split("val", args.val_json, args.val_images)

    # Cross-split checks
    train_json_set = set(iter_coco_image_filenames(args.train_json))
    val_json_set = set(iter_coco_image_filenames(args.val_json))
    overlap_json = sorted(train_json_set & val_json_set)

    train_disk_set = set(list_image_files(args.train_images))
    val_disk_set = set(list_image_files(args.val_images))
    overlap_disk = sorted(train_disk_set & val_disk_set)

    print("=== Summary ===")
    for s in (train, val):
        print(
            f"{s['split']}: json_unique={s['json_unique']} (total={s['json_total']}), "
            f"disk_unique={s['disk_unique']} (total={s['disk_total']})"
        )

    print("\n=== Train checks ===")
    print_report("Missing train JSON images on disk", train["missing_on_disk"], args.max_list)
    print_report("Extra train images on disk not in train JSON", train["extra_on_disk"], args.max_list)
    print(f"Duplicate file_name entries in train JSON: {len(train['dup_in_json'])}")

    print("\n=== Val checks ===")
    print_report("Missing val JSON images on disk", val["missing_on_disk"], args.max_list)
    print_report("Extra val images on disk not in val JSON", val["extra_on_disk"], args.max_list)
    print(f"Duplicate file_name entries in val JSON: {len(val['dup_in_json'])}")

    print("\n=== Cross-split checks ===")
    print_report("Overlapping file_names between train.json and val.json", overlap_json, args.max_list)
    print_report("Overlapping filenames between images/train and images/val", overlap_disk, args.max_list)

    # Common “wrong split” symptom: train JSON references val disk files and vice versa.
    train_references_val_disk = sorted(train_json_set & val_disk_set)
    val_references_train_disk = sorted(val_json_set & train_disk_set)
    print_report(
        "Train JSON file_names that exist in images/val",
        train_references_val_disk,
        args.max_list,
    )
    print_report(
        "Val JSON file_names that exist in images/train",
        val_references_train_disk,
        args.max_list,
    )

    # Non-zero exit if any mismatch
    problems = (
        len(train["missing_on_disk"]) + len(val["missing_on_disk"]) + len(overlap_json) +
        len(train_references_val_disk) + len(val_references_train_disk)
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
