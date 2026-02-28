# Boeing Capstone: UVSD Enrichment Data Pipeline

## 📋 Project Overview
This repository contains the code and documentation for the **Urban Vehicle Scene Detection (UVSD) Enrichment Pipeline**, developed as part of the Boeing Capstone project. The system processes aerial imagery to detect and enrich vehicle instances using advanced vision-language models (InternVL2-26B) on the Kamiak HPC cluster.

## 🚀 Key Features
- **Vehicle Detection**: Processes high-resolution aerial imagery using YOLO-based detection
- **Attribute Enrichment**: Enhances vehicle detections with color, angle, description, and time-of-day attributes using InternVL2-26B
- **Batch Processing**: Optimized for HPC environments with SLURM job scheduling
- **Data Validation**: Comprehensive validation pipeline ensuring data integrity
- **Documentation**: Complete system architecture, data flow diagrams, and data dictionaries

## 📁 Repository Structure
```
UVSD_Enrichment/
├── Documentation/                    # Comprehensive project documentation
│   ├── PROJECT_SUMMARY.md           # Project overview and objectives
│   ├── architecture/                # System architecture diagrams
│   │   ├── high_level_architecture.md    # Mermaid.js architecture diagram
│   │   └── system_architecture_diagram.md # Detailed system architecture
│   ├── data_dictionary/             # Data schema and relationships
│   └── data_flow/                   # Data flow diagrams and processes
├── check_instances_vs_images.py     # Data validation script
├── download_model.py                # Model downloading utility
├── internvl_inference.py           # Main inference pipeline
├── uvsdjeffreyflow.py              # Workflow orchestration
├── run_internvl.sbatch             # SLURM batch job script
├── data_structure.md               # Dataset structure documentation
├── DATA_PREPARATION.md             # Data preparation guide
├── preflight_checks.md             # System verification checklist
└── Success_Formula                  # Success criteria document
```

## 🗺️ System Architecture

### Data Flow Diagram
The system follows this pipeline:
```
Raw Aerial Imagery → Tiling (640x640) → YOLO Detection → COCO Conversion → 
Data Validation → InternVL Inference → Attribute Enrichment → Results Storage
```

### Key Components
1. **Data Ingestion**: Processes GeoTIFF/JPEG aerial imagery into 640x640 tiles
2. **Model Inference**: Uses InternVL2-26B vision-language model for vehicle attribute enrichment
3. **Batch Processing**: SLURM-based job scheduling on Kamiak HPC cluster
4. **Validation**: Comprehensive checks for data integrity and model outputs

## 📊 Diagrams in This Repository

### 1. **High-Level Architecture Diagram** (`Documentation/architecture/high_level_architecture.md`)
Mermaid.js diagram showing the complete system architecture with:
- Data ingestion layer
- Processing layer with model inference
- Storage layer for results
- Infrastructure layer (Kamiak HPC)
- Monitoring and alerting systems

### 2. **Data Flow Diagram** (`Documentation/data_flow/data_flow_diagram.md`)
Mermaid.js flowchart illustrating:
- End-to-end data processing pipeline
- Transformation steps from raw images to enriched results
- Quality control checkpoints
- Batch processing workflow

### 3. **System Architecture Diagram** (`Documentation/architecture/system_architecture_diagram.md`)
Detailed component diagrams including:
- Component interaction diagrams
- Data flow sequence diagrams
- Geographic partitioning strategies

### 4. **Simple & Complete Diagrams** (`Documentation/uvsd_simple_diagram.md`, `Documentation/uvsd_complete_diagram.md`)
Additional Mermaid.js diagrams providing:
- Simplified overview for quick understanding
- Comprehensive view with all system components

## ⚠️ Large Files on Kamiak (Not Included in Repository)

**Important**: This GitHub repository contains only code and documentation. The following large files reside on the **Kamiak HPC cluster** and are excluded from this repository via `.gitignore`:

### Excluded Large Files/Directories:
- **`models/`** (48GB): Contains the InternVL2-26B model weights
- **`UVSD_Tiled_640/`**: Tiled image dataset directory
- **`instances_train.json`** (399MB): COCO format training annotations
- **`instances_val.json`** (100MB): COCO format validation annotations
- **`logs/`**: Execution logs and performance metrics
- **`offload_folder/`**: Model offloading directory
- **`results/`**: Generated analysis results
- **`.cache/`**: System cache files

### Accessing Large Files on Kamiak:
```bash
# On Kamiak HPC cluster:
cd /data/lab/lapin/UVSD_Enrichment/

# Large files available locally:
ls -lh models/                    # 48GB model directory
ls -lh instances_train.json       # 399MB training annotations
ls -lh instances_val.json         # 100MB validation annotations
```

## 🛠️ Setup and Usage

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- Access to Kamiak HPC cluster
- InternVL2-26B model weights (available on Kamiak)

### Installation
```bash
# Clone repository
git clone https://github.com/aggressor-FZX/Boeing_Capstone_Data_Pipeline_One.git
cd Boeing_Capstone_Data_Pipeline_One

# Install dependencies (example - adjust based on requirements.txt)
pip install torch torchvision transformers pillow numpy pandas
```

### Running the Pipeline
```bash
# Data validation
python check_instances_vs_images.py

# Model inference (on Kamiak)
sbatch run_internvl.sbatch

# Or run locally (with GPU)
python internvl_inference.py
```

## 📈 Results and Outputs
The pipeline generates enriched vehicle attributes including:
- Vehicle color classification
- Viewing angle estimation
- Natural language descriptions
- Time-of-day classification
- Confidence scores for each attribute

Outputs are stored in CSV format for downstream analysis and visualization.

## 🔗 Related Documentation
- **Data Dictionary**: `Documentation/data_dictionary/` - Complete schema documentation
- **Project Summary**: `Documentation/PROJECT_SUMMARY.md` - Project objectives and scope
- **Data Preparation**: `DATA_PREPARATION.md` - Dataset preparation guidelines
- **Preflight Checks**: `preflight_checks.md` - System verification steps

## 👥 Contributors
- **Jeffrey Calderon** - Primary developer and system architect
- **Boeing Capstone Team** - Project guidance and requirements

## 📄 License
This project is developed for academic and research purposes as part of the Boeing Capstone project.

## 🤝 Acknowledgments
- **Washington State University** for Kamiak HPC resources
- **Boeing Company** for project sponsorship and guidance
- **InternVL Team** for the vision-language model

---

*Note: For access to the large dataset files or model weights, please contact the project maintainers or access the Kamiak HPC cluster at `/data/lab/lapin/UVSD_Enrichment/`.*