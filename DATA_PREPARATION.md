# Data preparation for InternVL instance-segmentation (UVSD)

Purpose
- Explain, step-by-step, how to prepare image + annotation data so the project can "enrich" vehicle instances and run `internvl_inference.py` successfully.

Scope
- Input formats: COCO-style JSON (required), YOLO label files (common source), binary masks (SAM/Mask R-CNN).
- Output: validated `instances_train.json` / `instances_val.json` with polygon segmentations and matching image tiles under `UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/{train,val}`.

Summary (quick) ✅
- Images: JPG/PNG under `UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/{train,val}`
- Annotations: COCO `instances_*.json` in `UVSD_Enrichment/`
- REQUIRED: `annotation['segmentation'][0]` must be a flat polygon list [x1,y1,x2,y2,...]. The inference script explicitly reads `segmentation[0]` and will skip instances with empty/RLE segmentations.

Why polygons?
- `internvl_inference.py` extracts crops using the polygon in `annotation['segmentation'][0]` (it reshapes a flat list to Nx2). If you only have bboxes or RLE masks, convert them to polygons first.

Directory layout (what this repo expects)
- `UVSD_Enrichment/`
  - `instances_train.json`, `instances_val.json`  <-- COCO-format annotation files used by scripts
  - `internvl_inference.py`  <-- inference expects polygon segmentation
  - `check_instances_vs_images.py`  <-- validation helper
- `UVSD_Tiled_640/YOLO_Dataset_Tiled_640/`
  - `images/{train,val}`  <-- tiled images (filenames must match `images[].file_name` in COCO)
  - `labels/{train,val}`  <-- YOLO .txt (optional source)

Required COCO fields (must be present and consistent)
- images[]: `id` (int), `file_name` (string), `width` (int), `height` (int)
- annotations[]: `id`, `image_id` (matches `images.id`), `category_id`, `segmentation` (polygon list), `bbox` ([x,y,w,h]), `area`, `iscrowd`
- categories[]: include mapping of `id`→`name` (e.g. `vehicle`)

Segmentation format details (critical)
- `segmentation` must be a list whose first element is a flat polygon list: `[x1,y1,x2,y2,...]`.
- Minimum polygon: 3 points → 6 numbers. Very small / invalid polygons will be skipped.
- RLE masks: convert to polygon before using (see conversion snippet below).
- Coordinates: pixel coordinates in the same coordinate frame as the tile image file referenced by `images[].file_name`.

Important script behavior to match
- `internvl_inference.py` does:
  - `seg = annotation['segmentation'][0]`
  - `pts = np.array(seg).reshape(-1,2).astype(np.int32)`
  - If `segmentation` is empty or missing, that instance is skipped.
  - Crops are created from padded `bbox` with pad=20px; very small crops are ignored.

Common sources → conversion recipes
1) YOLO (.txt) → COCO polygon (quick rectangle)
- YOLO format: `class x_center y_center width height` (normalized). Convert to pixel bbox and make a 4-point rectangle polygon.
- Example (Python):

```python
# yolo_line: class_id x_c y_c w h  (all normalized)
def yolo_to_coco_bbox(yolo_line, img_w, img_h):
    cls, x_c, y_c, w, h = map(float, yolo_line.split())
    x_c *= img_w; y_c *= img_h; w *= img_w; h *= img_h
    x = x_c - w/2; y = y_c - h/2
    return [x, y, w, h]

# convert bbox -> polygon (flat list)
def bbox_to_polygon(bbox):
    x,y,w,h = bbox
    return [x,y, x+w,y, x+w,y+h, x,y+h]
```

Note: rectangle polygons from boxes are acceptable for coarse masks but quality is lower than mask-based polygons.

2) Binary mask (SAM / Mask R-CNN / png) → COCO polygon
- Use OpenCV `findContours` or `pycocotools` to extract contours and flatten to `[x1,y1,...]`.
- Example (Python):

```python
import cv2
import numpy as np
from pycocotools import mask as maskUtils

# If you have RLE (pycocotools rle):
# binary_mask = maskUtils.decode(rle)
# Otherwise load PNG mask as binary_mask (H x W) boolean/uint8

contours, _ = cv2.findContours(binary_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise ValueError('no contours found')
# choose largest contour
cnt = max(contours, key=cv2.contourArea)
poly = cnt.reshape(-1, 2).tolist()
segmentation = [sum(poly, [])]  # flatten and wrap in list

# compute bbox & area
x,y,w,h = cv2.boundingRect(cnt)
area = float(maskUtils.area(maskUtils.encode(np.asfortranarray(binary_mask.astype('uint8')))))
```

Helpful tip: use `cv2.approxPolyDP` to reduce polygon vertex count if needed.

Validation & checks (run these before inference)
- 1) File-name vs JSON checks:
  - Command:
    ```bash
    python check_instances_vs_images.py \
      --train-json instances_train.json --val-json instances_val.json \
      --train-images UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/train \
      --val-images   UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/val
    ```
  - This will report missing files, duplicates and cross-split leaks.

- 2) Ensure every `annotation` you intend to process has a non-empty polygon in `segmentation[0]`.
  - Quick Python sanity check snippet:

```python
import json
with open('instances_val.json') as f:
    data = json.load(f)
for ann in data['annotations']:
    seg = ann.get('segmentation')
    if not seg or not seg[0] or len(seg[0]) < 6:
        print('bad', ann['id'])
```

- 3) Ensure `images[].file_name` exactly matches tile filenames under `images/val` or `images/train`.

- 4) Bounding boxes and polygon coords must lie within `[0, width]` and `[0, height]` of the corresponding image.

How to test inference locally (quick run)
- Edit `internvl_inference.py` configuration near the top:
  - `TEST_MODE = True`
  - `TEST_COUNT_IMAGES = 50` (or desired small number)
  - `TEST_COUNT_VEHICLES_PER_IMAGE = 2`
- Run:
  ```bash
  python internvl_inference.py
  ```
- Output CSV: `results/vehicle_instance_analysis_val.csv`

SLURM / full run
- Script: `run_internvl.sbatch` is provided. Confirm environment variables and `HF_HOME` paths are correct, then submit with `sbatch run_internvl.sbatch`.

Quality and dataset tips
- Prefer mask-derived polygons over bbox approximations for best model input.
- Remove tiny/noisy annotations (area < threshold) — `internvl_inference.py` may skip very small crops.
- Simplify very long polygons with `approxPolyDP` to reduce downstream processing cost.
- Keep category ids consistent and include `categories[]` in the COCO file.

Troubleshooting (common problems)
- "Skipped instances / zero segmentation": segmentation is empty or RLE — convert to polygon.
- "File NOT found": `images[].file_name` doesn't match disk — run `check_instances_vs_images.py`.
- "Tiny crops ignored": increase annotation bbox size or filter out tiny annotations.
- "Coordinate mismatch (global vs tile)": ensure annotations are in tile pixel coordinates (the same image file that `images[].file_name` points to).

Quick checklist before submitting a training/inference job ✅
1. All image files present under `UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/{train,val}`
2. `instances_train.json` / `instances_val.json` point to those filenames and contain `segmentation[0]` polygons
3. `bbox` and `area` fields present and accurate
4. Run `python check_instances_vs_images.py` → zero missing/duplicate problems
5. Run `internvl_inference.py` in `TEST_MODE` to validate outputs

Appendix — example COCO annotation (minimal)
```json
{
  "images": [ { "id": 123, "file_name": "val/000123_tile_0_0.jpg", "width": 640, "height": 640 } ],
  "annotations": [ {
      "id": 1,
      "image_id": 123,
      "category_id": 1,
      "segmentation": [[100.0,50.0, 150.0,50.0, 150.0,90.0, 100.0,90.0]],
      "bbox": [100.0,50.0,50.0,40.0],
      "area": 2000.0,
      "iscrowd": 0
  } ],
  "categories": [ {"id":1,"name":"vehicle"} ]
}
```

Need a conversion script? 🔧
- I can add: (A) YOLO→COCO converter, (B) mask→polygon converter, or (C) automatic SAM-based polygon generation for all YOLO boxes — tell me which and I will add it to `UVSD_Enrichment/`.

---
Last updated: 2026-02-17
Files referenced: `internvl_inference.py`, `check_instances_vs_images.py`, `instances_train.json`, `instances_val.json`
