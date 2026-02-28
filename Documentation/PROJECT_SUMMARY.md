# UVSD Enrichment Project - Complete Documentation

## Project Overview
The UVSD (Urban Vehicle Scene Detection) Enrichment project is a comprehensive computer vision pipeline for processing aerial imagery to detect, segment, and enrich vehicle instances with descriptive attributes using the InternVL2 vision-language model.

## Documentation Structure

### 1. Data Flow Documentation (`/Documentation/data_flow/`)
- **`data_flow_diagram.md`**: Mermaid diagram showing end-to-end data flow
- **`visual_data_flow.txt`**: ASCII art representation of the pipeline
- **Purpose**: Visualize how data moves through the system from ingestion to results

### 2. Data Dictionary (`/Documentation/data_dictionary/`)
- **`images_table.csv`**: Metadata for tiled images
- **`annotations_table.csv`**: Vehicle detection annotations
- **`categories_table.csv`**: Vehicle category definitions
- **`enriched_results_table.csv`**: Model-enriched attributes
- **`README.md`**: Overview of data dictionary structure
- **`DATA_DICTIONARY_SUMMARY.md`**: Complete data dictionary with business rules
- **Purpose**: Define all data entities, relationships, and metadata

### 3. Architecture Documentation (`/Documentation/architecture/`)
- **`high_level_architecture.md`**: Detailed system architecture
- **`system_architecture_diagram.md`**: Comprehensive architecture diagrams
- **Purpose**: Document system design, components, and interactions

### 4. Project README (`/Documentation/README.md`)
- **Complete project overview**
- **Data source and characteristics**
- **Data wrangling techniques**
- **Storage strategy**
- **Usage instructions**
- **Purpose**: High-level project documentation

## Key Features Documented

### Data Pipeline
1. **Image Tiling**: Converts large aerial images to 640x640 tiles
2. **Annotation Conversion**: Transforms YOLO format to COCO JSON
3. **Data Validation**: Cross-checks images and annotations
4. **Model Inference**: Runs InternVL2 for attribute enrichment
5. **Results Storage**: Saves enriched data to CSV format

### System Architecture
1. **Batch Processing**: SLURM-based HPC execution
2. **GPU Acceleration**: Utilizes NVIDIA GPUs for model inference
3. **Quality Assurance**: Multiple validation checkpoints
4. **Scalable Design**: Supports horizontal and vertical scaling

### Data Management
1. **Structured Storage**: Organized directory hierarchy
2. **Metadata Tracking**: Comprehensive data dictionary
3. **Quality Metrics**: Validation and accuracy measurements
4. **Version Control**: Schema evolution and data versioning

## Data Specifications

### Volume
- **Images**: ~16,000 tiles (640x640 pixels each)
- **Annotations**: ~16,000 vehicle instances
- **Enriched Results**: ~16,000 records with 12+ attributes each
- **Model Size**: ~50GB (InternVL2-5-26B)
- **Total Storage**: ~55GB (including model weights)

### Formats
- **Images**: JPEG format
- **Annotations**: COCO JSON format
- **Results**: CSV format
- **Configurations**: YAML format
- **Logs**: Text and JSON formats

### Quality Standards
- **Accuracy**: >90% vehicle detection accuracy
- **Completeness**: 100% annotation-image matching
- **Consistency**: Uniform formatting across all files
- **Timeliness**: Batch processing within 24 hours

## Technical Stack

### Core Technologies
- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch
- **Vision-Language Model**: InternVL2-5-26B
- **HPC Scheduler**: SLURM
- **Data Processing**: Pandas, NumPy, OpenCV

### Infrastructure
- **Compute**: Kamiak HPC Cluster with GPU nodes
- **Storage**: Parallel filesystem for high I/O
- **Networking**: High-speed interconnects
- **Containerization**: Docker/Singularity for reproducibility

### Monitoring & Logging
- **Log Management**: Structured logging to `logs/` directory
- **Performance Metrics**: GPU utilization, throughput, latency
- **Alerting**: Email/Slack notifications for failures
- **Visualization**: Custom dashboards for monitoring

## Usage Scenarios

### For Data Scientists
1. **Data Exploration**: Use CSV files for statistical analysis
2. **Model Training**: Leverage enriched data for downstream tasks
3. **Quality Assessment**: Validate model outputs against ground truth
4. **Research Publication**: Cite dataset in academic papers

### For Developers
1. **Pipeline Extension**: Add new processing stages
2. **Model Integration**: Incorporate new vision-language models
3. **Performance Optimization**: Improve throughput and latency
4. **API Development**: Create REST APIs for data access

### For System Administrators
1. **Cluster Management**: Monitor resource utilization
2. **Storage Management**: Optimize data layout and access patterns
3. **Security Hardening**: Implement access controls and encryption
4. **Backup Strategy**: Design data backup and recovery procedures

## Future Roadmap

### Short-term (Next 3 Months)
1. **Streaming Pipeline**: Real-time processing capability
2. **Model Optimization**: Quantization for faster inference
3. **API Development**: REST API for data access
4. **Dashboard Creation**: Web-based monitoring interface

### Medium-term (Next 6 Months)
1. **Additional Attributes**: Vehicle make/model detection
2. **Geospatial Integration**: GPS coordinates and map overlays
3. **Temporal Analysis**: Vehicle movement tracking over time
4. **Multi-modal Fusion**: Combine with other sensor data

### Long-term (Next 12 Months)
1. **Edge Deployment**: On-device inference capability
2. **Federated Learning**: Privacy-preserving model training
3. **Automated Quality Control**: ML-based validation pipeline
4. **Community Dataset**: Open-source release of enriched data

## Getting Started

### Prerequisites
1. **Access to Kamiak HPC Cluster**
2. **Python 3.8+ with required packages**
3. **InternVL2 model weights (~50GB)**
4. **Aerial imagery dataset**

### Quick Start
1. **Prepare Data**: Place images in `UVSD_Tiled_640/YOLO_Dataset_Tiled_640/images/`
2. **Convert Annotations**: Convert YOLO labels to COCO JSON format
3. **Validate Data**: Run `check_instances_vs_images.py`
4. **Run Inference**: Submit `run_internvl.sbatch` to SLURM
5. **Analyze Results**: Examine `results/vehicle_instance_analysis_val.csv`

### Common Workflows
1. **Batch Processing**: Process large datasets overnight
2. **Incremental Updates**: Add new images to existing dataset
3. **Quality Checks**: Validate annotations before inference
4. **Performance Tuning**: Optimize batch sizes and GPU settings

## Support and Contact

### Documentation
- **Data Dictionary**: `/Documentation/data_dictionary/`
- **Architecture**: `/Documentation/architecture/`
- **Data Flow**: `/Documentation/data_flow/`
- **README**: `/Documentation/README.md`

### Code Repository
- **Main Scripts**: Root directory Python files
- **Configuration**: `dataset.yaml` and SLURM scripts
- **Utilities**: Validation and conversion scripts
- **Examples**: Sample data and configuration files

### Troubleshooting
1. **Data Issues**: Check `preflight_checks.md`
2. **Model Issues**: Review `logs/` directory
3. **Performance Issues**: Monitor GPU utilization
4. **Quality Issues**: Examine `bbox_checks/` visualizations

### Contact Information
- **Project Lead**: [Your Name/Team]
- **Documentation Maintainer**: [Your Name/Team]
- **Last Updated**: February 27, 2026
- **Version**: 1.0

## License and Attribution
- **Data License**: [Specify license for your data]
- **Code License**: [Specify license for your code]
- **Model License**: Check InternVL2 license terms
- **Citation**: Please cite this work if used in research

---

*This documentation was automatically generated as part of the UVSD Enrichment project documentation initiative.*