# UVSD Enrichment Project Documentation

## Project Overview
The UVSD (Urban Vehicle Scene Detection) Enrichment project is a computer vision pipeline that processes aerial/satellite imagery to detect, segment, and enrich vehicle instances with descriptive attributes using a vision-language model (InternVL2).

## Data Source and Characteristics

### Original Data Source
- **Type**: High-resolution aerial/satellite imagery
- **Format**: Large geospatial image files
- **Content**: Urban scenes with vehicles
- **Resolution**: Variable spatial resolution (typically 0.15-0.30 meters per pixel)

### Processed Dataset
- **Tiling**: Original images divided into 640x640 pixel tiles
- **Format**: YOLO-style dataset structure
- **Splits**: Training (80%) and Validation (20%) sets
- **Total Images**: Approximately 16,000+ tiles
- **Total Annotations**: Approximately 16,000+ vehicle instances

### Annotation Format
- **Original**: YOLO format (.txt files with normalized coordinates)
- **Processing**: Converted to COCO JSON format
- **Current**: `instances_train.json` and `instances_val.json`
- **Attributes**: Bounding boxes, segmentation polygons, area calculations

## Special Properties

### Unique Characteristics
1. **Tiled Processing**: Large images processed as smaller tiles for computational efficiency
2. **Polygon Segmentation**: Detailed vehicle outlines rather than just bounding boxes
3. **Vision-Language Integration**: Combines computer vision with natural language understanding
4. **Attribute Enrichment**: Adds semantic attributes (color, angle, description, time of day)

### Data Challenges
1. **Scale Variation**: Vehicles appear at different scales based on altitude
2. **Occlusion**: Partial visibility of vehicles in urban scenes
3. **Lighting Conditions**: Variable illumination (daytime, night, shadows)
4. **Viewing Angles**: Multiple perspectives (top-down, oblique, side views)

## Data Wrangling Techniques

### Preprocessing Pipeline
1. **Image Tiling**
   - Input: Large geospatial images
   - Process: Sliding window 640x640 with overlap
   - Output: Tiled image dataset

2. **Annotation Conversion**
   - Input: YOLO format labels
   - Process: Normalized → pixel coordinate conversion
   - Output: COCO JSON format annotations

3. **Data Validation**
   - Cross-check: Images vs. annotations matching
   - Quality control: Visual bounding box verification
   - Format validation: Polygon integrity checks

4. **Model Inference**
   - Input: Validated images + annotations
   - Process: InternVL2 vision-language model
   - Output: Enriched vehicle attributes

### Storage Strategy
- **Raw Data**: Compressed archives (UVSD_Tiled_640.zip)
- **Processed Data**: Hierarchical directory structure
- **Annotations**: JSON files with COCO format
- **Results**: CSV files with enriched attributes
- **Models**: Local storage of InternVL2 weights (~26B parameters)

## Proposed Storage Architecture

### Tier 1: Raw Storage
- **Format**: Compressed ZIP archives
- **Location**: Project root directory
- **Purpose**: Long-term archival of original data

### Tier 2: Processed Storage
- **Format**: Organized directory structure
- **Location**: `UVSD_Tiled_640/YOLO_Dataset_Tiled_640/`
- **Subdirectories**: `images/{train,val}`, `labels/{train,val}`
- **Purpose**: Active dataset for model training/inference

### Tier 3: Annotation Storage
- **Format**: JSON files (COCO format)
- **Location**: Project root directory
- **Files**: `instances_train.json`, `instances_val.json`
- **Purpose**: Standardized annotation format for compatibility

### Tier 4: Results Storage
- **Format**: CSV files with enriched data
- **Location**: `results/` directory
- **Files**: `vehicle_instance_analysis_val.csv`
- **Purpose**: Model outputs for analysis and reporting

### Tier 5: Model Storage
- **Format**: Safetensors + configuration files
- **Location**: `models/InternVL2_5-26B/`
- **Size**: ~50GB (11 shards + index)
- **Purpose**: Local model weights for inference

## Data Volume Estimates

### Current Dataset
- **Images**: ~16,000 tiles (640x640 pixels each)
- **Storage**: ~2-3GB (compressed JPEG)
- **Annotations**: ~16,000 vehicle instances
- **Metadata**: ~100MB (JSON + CSV files)
- **Model**: ~50GB (InternVL2 weights)

### Scalability
- **Horizontal Scaling**: Add more source imagery
- **Vertical Scaling**: Increase tile resolution
- **Attribute Expansion**: Add more enrichment categories
- **Temporal Dimension**: Add time-series data

## Quality Assurance

### Validation Procedures
1. **Instance-Image Matching**: Ensure each annotation has corresponding image
2. **Bounding Box Accuracy**: Visual verification through `bbox_checks/`
3. **Segmentation Integrity**: Polygon coordinate validation
4. **Model Output Consistency**: Cross-validation of enriched attributes

### Monitoring Metrics
- **Completeness**: Percentage of annotations with valid enrichments
- **Accuracy**: Comparison with ground truth (if available)
- **Consistency**: Uniform formatting across all data files
- **Timeliness**: Processing time per batch

## Future Enhancements

### Data Expansion
1. **Additional Vehicle Types**: Classify by vehicle category (car, truck, bus, etc.)
2. **Geospatial Context**: Add GPS coordinates and map integration
3. **Temporal Analysis**: Track vehicle movements over time
4. **Weather Conditions**: Incorporate weather data for context

### Technical Improvements
1. **Streaming Pipeline**: Real-time processing of incoming imagery
2. **Distributed Processing**: Scale across multiple GPU nodes
3. **Automated Quality Control**: ML-based validation of annotations
4. **Interactive Visualization**: Web-based data exploration tools

## Usage Instructions

### For Data Scientists
1. Access processed data in `UVSD_Tiled_640/` directory
2. Use `instances_*.json` for COCO-compatible annotations
3. Analyze enriched results in `results/vehicle_instance_analysis_val.csv`
4. Refer to data dictionary for field definitions

### For Developers
1. Follow data flow diagram for pipeline understanding
2. Use `check_instances_vs_images.py` for validation
3. Modify `internvl_inference.py` for custom processing
4. Review `run_internvl.sbatch` for HPC execution

### For Researchers
1. Cite original data sources appropriately
2. Document any modifications to the pipeline
3. Share enriched datasets with proper attribution
4. Contribute back improvements to the community

## Contact and Support
- **Project Maintainer**: [Your Name/Team]
- **Documentation Version**: 1.0
- **Last Updated**: February 27, 2026
- **Repository**: UVSD_Enrichment GitLab Project