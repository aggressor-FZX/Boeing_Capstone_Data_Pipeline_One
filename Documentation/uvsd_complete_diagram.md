# UVSD Enrichment Complete System Diagram

## Overview
This diagram provides a comprehensive visualization of the UVSD Enrichment pipeline architecture, data flow, and system components.

## Complete System Architecture Diagram

```mermaid
flowchart TD
    %% ========== DATA SOURCES LAYER ==========
    subgraph DS["Data Sources Layer"]
        direction LR
        A1[Raw Aerial Imagery<br/>High-resolution satellite/airborne]
        A2[YOLO Annotation Files<br/>.txt format with bounding boxes]
    end
    
    %% ========== PREPROCESSING LAYER ==========
    subgraph PL["Preprocessing Layer"]
        direction TB
        B1[Tiling Service<br/>640x640 pixel tiles]
        B2[Format Conversion<br/>YOLO → COCO]
        B3[Data Validation<br/>check_instances_vs_images.py]
        
        A1 --> B1
        A2 --> B2
        B1 --> B3
        B2 --> B3
    end
    
    %% ========== STORAGE LAYER ==========
    subgraph SL["Storage Layer"]
        direction LR
        C1[Tiled Images<br/>UVSD_Tiled_640/images/]
        C2[COCO Annotations<br/>instances_train.json<br/>instances_val.json]
        C3[Validated Dataset<br/>Ready for processing]
        
        B1 --> C1
        B2 --> C2
        B3 --> C3
    end
    
    %% ========== PROCESSING LAYER ==========
    subgraph PRL["Processing Layer"]
        direction TB
        D1[Batch Job Submission<br/>run_internvl.sbatch]
        D2[GPU Cluster Execution<br/>Kamiak HPC System]
        D3[Model Inference<br/>internvl_inference.py]
        D4[Vision-Language Model<br/>InternVL2_5-26B]
        D5[Attribute Enrichment<br/>Color, Angle, Description, Time]
        
        C3 --> D1
        D1 --> D2
        D2 --> D3
        D3 --> D4
        D4 --> D5
    end
    
    %% ========== OUTPUT LAYER ==========
    subgraph OL["Output Layer"]
        direction LR
        E1[Enriched Results CSV<br/>vehicle_instance_analysis_val.csv]
        E2[Statistical Analysis<br/>Data insights & reporting]
        E3[Visual Validation<br/>bbox_checks/ directory]
        E4[Documentation<br/>Project reports & diagrams]
        
        D5 --> E1
        E1 --> E2
        B3 --> E3
    end
    
    %% ========== INFRASTRUCTURE LAYER ==========
    subgraph IL["Infrastructure Layer"]
        direction TB
        F1[HPC Cluster<br/>Multiple GPU nodes]
        F2[SLURM Scheduler<br/>Job management]
        F3[Storage Systems<br/>Weka filesystem]
        F4[Monitoring<br/>logs/ directory]
    end
    
    %% ========== COMPONENT INTERACTIONS ==========
    subgraph CI["Component Interactions"]
        direction LR
        G1[Python Scripts<br/>uvsdjeffreyflow.py<br/>download_model.py]
        G2[Configuration Files<br/>preflight_checks.md<br/>data_structure.md]
        G3[Documentation<br/>PROJECT_SUMMARY.md<br/>DATA_PREPARATION.md]
    end
    
    %% ========== CONNECTIONS BETWEEN LAYERS ==========
    SL --> PRL
    PRL --> OL
    IL --> PRL
    CI --> PRL
    
    %% ========== STYLING ==========
    linkStyle 0 stroke:#ff6b6b,stroke-width:3px
    linkStyle 1 stroke:#4ecdc4,stroke-width:3px
    linkStyle 2 stroke:#45b7d1,stroke-width:3px
    linkStyle 3 stroke:#96ceb4,stroke-width:3px
    linkStyle 4 stroke:#feca57,stroke-width:3px
    linkStyle 5 stroke:#ff9ff3,stroke-width:3px
    linkStyle 6 stroke:#54a0ff,stroke-width:3px
    linkStyle 7 stroke:#5f27cd,stroke-width:3px
    linkStyle 8 stroke:#00d2d3,stroke-width:3px
    linkStyle 9 stroke:#ff9f43,stroke-width:3px
    
    classDef dataSource fill:#c8d6e5,stroke:#1dd1a1,stroke-width:2px
    classDef preprocessing fill:#ff9ff3,stroke:#f368e0,stroke-width:2px
    classDef storage fill:#54a0ff,stroke:#2e86de,stroke-width:2px
    classDef processing fill:#ff6b6b,stroke:#ee5a24,stroke-width:2px
    classDef output fill:#10ac84,stroke:#1dd1a1,stroke-width:2px
    classDef infrastructure fill:#feca57,stroke:#ff9f43,stroke-width:2px
    classDef components fill:#5f27cd,stroke:#341f97,stroke-width:2px
    
    class A1,A2 dataSource
    class B1,B2,B3 preprocessing
    class C1,C2,C3 storage
    class D1,D2,D3,D4,D5 processing
    class E1,E2,E3,E4 output
    class F1,F2,F3,F4 infrastructure
    class G1,G2,G3 components
```

## Detailed Component Description

### Data Sources Layer
- **Raw Aerial Imagery**: High-resolution satellite/airborne images
- **YOLO Annotation Files**: Pre-labeled bounding boxes in YOLO format

### Preprocessing Layer
1. **Tiling Service**: Converts large images to 640x640 pixel tiles
2. **Format Conversion**: Transforms YOLO annotations to COCO format
3. **Data Validation**: Validates image-annotation consistency

### Storage Layer
1. **Tiled Images**: Organized in train/val directories
2. **COCO Annotations**: JSON files with instance data
3. **Validated Dataset**: Quality-checked ready dataset

### Processing Layer
1. **Batch Submission**: SLURM job scheduling
2. **GPU Execution**: HPC cluster with GPU acceleration
3. **Model Inference**: InternVL vision-language model
4. **Attribute Enrichment**: Color, angle, description, time analysis

### Output Layer
1. **Enriched Results**: CSV files with vehicle attributes
2. **Statistical Analysis**: Data insights and reporting
3. **Visual Validation**: Bounding box verification images
4. **Documentation**: Project reports and system diagrams

### Infrastructure Layer
- HPC Cluster with GPU nodes
- SLURM job scheduler
- Weka filesystem storage
- Comprehensive monitoring system

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant User as User/Researcher
    participant SLURM as SLURM Scheduler
    participant GPU as GPU Cluster
    participant Model as InternVL Model
    participant Storage as Results Storage
    
    User->>SLURM: Submit batch job<br/>run_internvl.sbatch
    SLURM->>GPU: Allocate resources
    GPU->>Model: Load InternVL2_5-26B
    Model->>GPU: Process tiled images
    GPU->>Storage: Save enriched results
    Storage->>User: Generate analysis reports
```

## Key Technical Specifications

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Image Tiling** | 640x640 pixels | Standardize input size |
| **Model** | InternVL2_5-26B | Vision-language inference |
| **GPU** | H100/A100/Tesla | Accelerated processing |
| **Storage** | Weka filesystem | High-performance storage |
| **Output Format** | CSV with attributes | Structured results |
| **Validation** | Bounding box checks | Quality assurance |

## Usage Instructions

1. **View in VS Code**: Install "Mermaid Preview" extension
2. **View on GitHub**: Diagrams render automatically
3. **Edit**: Modify the Mermaid code to update the diagram
4. **Export**: Use Mermaid Live Editor for PNG/SVG export

This diagram provides a complete visual representation of your UVSD Enrichment pipeline architecture.