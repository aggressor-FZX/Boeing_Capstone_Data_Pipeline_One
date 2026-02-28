# UVSD_Enrichment data layout

This document summarizes the current workspace data structure and what each directory contains.

## Top-level

- .cache/ - Cache data created by tools (not part of the dataset itself).
- UVSD_Tiled_640/ - Main YOLO-style tiled dataset directory.
- UVSD_Tiled_640.zip - Archive of the tiled dataset.
- bbox_checks/ - Visual checks of bounding boxes.
- logs/ - Job outputs and errors.
- models/ - Local model files and configs.
- offload_folder/ - Runtime offload storage (model or inference temp data).
- results/ - Analysis outputs and reports.
- __pycache__/ - Python bytecode cache.

Top-level files

- check_instances_vs_images.py - Script to validate dataset instances vs images.
- download_model.py - Script to download model assets.
- instances_train.json - COCO-style training instances.
- instances_val.json - COCO-style validation instances.
- internvl_inference.py - Inference script for InternVL.
- run_internvl.sbatch - Slurm batch script for InternVL runs.
- uvsdjeffreyflow.py - Project workflow / pipeline script.

## UVSD_Tiled_640/

- YOLO_Dataset_Tiled_640/ - YOLO dataset root.

### UVSD_Tiled_640/YOLO_Dataset_Tiled_640/

- dataset.yaml - YOLO dataset configuration.
- images/ - Image tiles.
- labels/ - YOLO label text files matching the tiles.

#### UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/

- train/ - Training image tiles.
- val/ - Validation image tiles.

#### UVSD_Tiled_640/YOLO_Dataset_Tiled_640/labels/

- train/ - Training labels for tiles (one .txt per image tile).
- val/ - Validation labels for tiles.

## bbox_checks/

- val_bbox_check_01075_tile_0_0.png - Example visual check image for bounding boxes.

## logs/

- internvl_20383319.err/.out - Slurm stderr/stdout from InternVL runs.
- internvl_20383422.err/.out - Additional run logs.
- internvl_20383449.err/.out - Additional run logs.
- internvl_20383465.err/.out - Additional run logs.
- internvl_20383508.err/.out - Additional run logs.

## models/

- InternVL2_5-26B/ - Local copy of the InternVL2.5 26B model and configs.

### models/InternVL2_5-26B/

- model-00001-of-00011.safetensors ... model-00011-of-00011.safetensors - Sharded model weights.
- model.safetensors.index.json - Index for sharded weights.
- config.json, generation_config.json - Model configuration.
- preprocessor_config.json, tokenizer_config.json, tokenizer.model - Tokenizer and preprocessing config.
- tokenization_internlm2.py, tokenization_internlm2_fast.py - Tokenizer code.
- modeling_intern_vit.py, modeling_internlm2.py, modeling_internvl_chat.py - Model code.
- configuration_intern_vit.py, configuration_internlm2.py, configuration_internvl_chat.py - Model configuration code.
- added_tokens.json, special_tokens_map.json - Token mappings.
- conversation.py - Chat formatting utilities.
- README.md - Model documentation.
- examples/ - Example inputs for the model (image1.jpg, image2.jpg, red-panda.mp4).
- runs/ - Model run artifacts.

### models/InternVL2_5-26B/runs/

- Nov19_11-48-37_HOST-10-140-60-39/ - Example run directory with event logs.

## results/

- vehicle_instance_analysis.csv - Analysis output file.

## offload_folder/

- Runtime offload data used during model loading or inference.
