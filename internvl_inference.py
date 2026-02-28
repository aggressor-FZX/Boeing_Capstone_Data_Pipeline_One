#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternVL Inference on Kamiak HPC - Multi-GPU Optimized
Analyzes individual segmented vehicles using multiple GPUs
"""


import torch
import sys
import os
import json
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import accelerate.big_modeling
from PIL import Image


# ============== CONFIGURATION ==============
BASE_DIR = "/data/lab/lapin/UVSD_Enrichment"
ANNOTATION_FILE = os.path.join(BASE_DIR, "instances_val.json")
IMAGE_DIR = os.path.join(BASE_DIR, "UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/val")
OUTPUT_CSV = os.path.join(BASE_DIR, "results/vehicle_instance_analysis_val.csv")
MODEL_PATH = '/data/lab/lapin/UVSD_Enrichment/models/InternVL2_5-26B'


# Test mode settings
TEST_MODE = True  # Set to False for full run
TEST_COUNT_IMAGES = 1000  # None for all images
TEST_COUNT_VEHICLES_PER_IMAGE = None  # None for all vehicles per image

# Multi-GPU optimized settings
NUM_GPUS = 1  # Number of GPUs to use (2, 4, etc.)
BATCH_SIZE = 4 # Increased for multi-GPU (adjust based on VRAM)
SAVE_INTERVAL = 50


# ============== SETUP ==============
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")


# Check GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"✅ Detected {num_gpus} GPUs:")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({vram:.2f} GB)")
    
    if num_gpus < NUM_GPUS:
        print(f"⚠️ Warning: Requested {NUM_GPUS} GPUs but only {num_gpus} available")
        NUM_GPUS = num_gpus
else:
    print("❌ No GPU detected!")
    sys.exit(1)


# Create output directories
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs("offload_folder", exist_ok=True)


# ============== HOTFIX FOR ACCELERATE ==============
_original_load = accelerate.big_modeling.load_checkpoint_and_dispatch


def patched_load_checkpoint_and_dispatch(model, checkpoint, **kwargs):
    if 'offload_folder' not in kwargs or kwargs['offload_folder'] is None:
        kwargs['offload_folder'] = "offload_folder"
    return _original_load(model, checkpoint, **kwargs)


accelerate.big_modeling.load_checkpoint_and_dispatch = patched_load_checkpoint_and_dispatch
print("🛠️ Applied patch to 'accelerate' for offload_folder.")


# ============== LOAD DATASET ==============
print(f"📂 Loading annotations from: {ANNOTATION_FILE}")
if not os.path.exists(ANNOTATION_FILE):
    print(f"❌ Error: Annotation file not found at {ANNOTATION_FILE}")
    sys.exit(1)


with open(ANNOTATION_FILE, 'r') as f:
    coco_data = json.load(f)


# Filter valid images
valid_images = []
for img in coco_data.get('images', []):
    fname = os.path.basename(img['file_name'])
    full_path = os.path.join(IMAGE_DIR, fname)
    if os.path.exists(full_path):
        valid_images.append({
            'id': img['id'],
            'path': full_path,
            'file_name': fname
        })


print(f"✅ Found {len(valid_images)} valid images in {IMAGE_DIR}")


# Group annotations by image_id
print("📋 Grouping annotations by image...")
img_to_anns = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_anns:
        img_to_anns[img_id] = []
    img_to_anns[img_id].append(ann)


# Apply test mode limits
if TEST_MODE and TEST_COUNT_IMAGES:
    valid_images = valid_images[:TEST_COUNT_IMAGES]
    print(f"⚠️ TEST MODE: Processing only {len(valid_images)} images")
    if TEST_COUNT_VEHICLES_PER_IMAGE:
        print(f"⚠️ TEST MODE: Max {TEST_COUNT_VEHICLES_PER_IMAGE} vehicles per image")


# Count total vehicle instances
total_vehicles = 0
for img in valid_images:
    anns = img_to_anns.get(img['id'], [])
    if TEST_MODE and TEST_COUNT_VEHICLES_PER_IMAGE:
        anns = anns[:TEST_COUNT_VEHICLES_PER_IMAGE]
    total_vehicles += len(anns)


print(f"🚗 Total vehicle instances to analyze: {total_vehicles}")


# ============== MODEL SETUP (MULTI-GPU) ==============
print(f"🚀 Initializing InternVL2.5 with {NUM_GPUS} GPUs...")
torch.cuda.empty_cache()


backend_config = TurbomindEngineConfig(
    session_len=4096,            # Reduce from 8704
    max_context_token_num=4096,  # Reduce from 8192
    cache_max_entry_count=0.9,
    tp=NUM_GPUS
)


gen_config = GenerationConfig(
    top_p=0.8,
    temperature=0.2,
    max_new_tokens=100,
    top_k=40,
    repetition_penalty=1.05
)
pipe = pipeline(MODEL_PATH, backend_config=backend_config, offload_folder="offload_folder")
print(f"✅ Model Loaded on {NUM_GPUS} GPUs!")


# ============== OPTIMIZED PROMPT ==============
prompt = (
    "Describe: vehicle color, angle, type, time of day. "
    "JSON: {\"color\": str, \"angle\": str, \"description\": str, \"time_of_day\": str}"
)



# ============== HELPER FUNCTION ==============
def extract_vehicle_crop(image_path, annotation):
    """Extract vehicle crop using segmentation polygon"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    if not annotation['segmentation'] or len(annotation['segmentation']) == 0:
        return None
    
    seg = annotation['segmentation'][0]
    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
    
    x, y, w, h = map(int, annotation['bbox'])
    
    # Add padding
    pad = 20
    x_pad = max(0, x - pad)
    y_pad = max(0, y - pad)
    w_pad = min(image.shape[1] - x_pad, w + 2*pad)
    h_pad = min(image.shape[0] - y_pad, h + 2*pad)
    
    crop = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
    
    # Skip tiny crops
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    
    return pil_image


# ============== INFERENCE LOOP ==============
results = []
vehicle_batch = []
ann_batch = []


print(f"🎥 Starting Multi-GPU Inference...")
print(f"   {total_vehicles} vehicles | Batch size: {BATCH_SIZE} | {NUM_GPUS} GPUs")


vehicle_count = 0
import time
start_time = time.time()


for img_info in tqdm(valid_images, desc="Images"):
    img_id = img_info['id']
    img_path = img_info['path']
    
    anns = img_to_anns.get(img_id, [])
    
    if TEST_MODE and TEST_COUNT_VEHICLES_PER_IMAGE:
        anns = anns[:TEST_COUNT_VEHICLES_PER_IMAGE]
    
    for ann in anns:
        vehicle_crop = extract_vehicle_crop(img_path, ann)
        
        if vehicle_crop is None:
            continue
        
        vehicle_batch.append(vehicle_crop)
        ann_batch.append({
            'annotation_id': ann['id'],
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'bbox': ann['bbox'],
            'area': ann['area']
        })
        
        # Process batch when full
        if len(vehicle_batch) >= 1:  # Process in smaller batches for stability
            try:
                inputs = [(prompt, img) for img in vehicle_batch]
                responses = pipe(inputs, gen_config=gen_config)
                
                for idx, resp in enumerate(responses):
                    results.append({
                        **ann_batch[idx],
                        "model_output": resp.text
                    })
                    vehicle_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing batch: {e}")
            
            # Clear batch
            vehicle_batch = []
            ann_batch = []
            
            # Progressive saving
            if vehicle_count % SAVE_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = vehicle_count / elapsed
                remaining = (total_vehicles - vehicle_count) / rate if rate > 0 else 0
                
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                print(f"\n💾 Checkpoint: {vehicle_count}/{total_vehicles} vehicles")
                print(f"   Speed: {rate:.1f} vehicles/sec | ETA: {remaining/60:.1f} min")


# Process remaining batch
if len(vehicle_batch) > 0:
    try:
        inputs = [(prompt, img) for img in vehicle_batch]
        responses = pipe(inputs, gen_config=gen_config)
        
        for idx, resp in enumerate(responses):
            results.append({
                **ann_batch[idx],
                "model_output": resp.text
            })
    except Exception as e:
        print(f"\n❌ Error processing final batch: {e}")


# ============== FINAL SAVE ==============
total_time = time.time() - start_time
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)


print(f"\n✅ Done! Analyzed {len(results)} vehicle instances")
print(f"   Results saved to: {OUTPUT_CSV}")
print(f"\n📊 Performance Statistics:")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Average speed: {len(results)/total_time:.2f} vehicles/second")
print(f"   Images processed: {len(valid_images)}")
if len(valid_images) > 0:
    print(f"   Avg vehicles per image: {len(results)/len(valid_images):.1f}")


print(f"\nFirst 5 results:")
print(df_results.head())
