# Keyence Analyzer - Architecture Overview

## System Architecture (WITH FILTERING--NO MORE FILTERING THOUGH)

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEYENCE ANALYZER SYSTEM                      │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   DATA LAYER (Eigen)                       │ │
│  │                                                            │ │
│  │  CSV File → Eigen::MatrixXf → Processing → Visualization   │ │
│  │             (master storage)                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │                           ▼                               │  │
│  │         ┌─────────────────────────────────────┐           │  │
│  │         │    PROCESSING PIPELINE              │           │  │
│  │         │                                     │           │  │
│  │         │  1. Load CSV                        │           │  │
│  │         │  2. Handle Invalid Points           │           │  │
│  │         │  3. Apply Filtering ◄── PCL Module  │           │  │
│  │         │  4. Downsample                      │           │  │
│  │         │  5. Visualize                       │           │  │
│  │         └─────────────────────────────────────┘           │  │
│  │                           │                               │  │
│  │         ┌─────────────────┴─────────────────┐             │  │
│  │         ▼                                   ▼             │  │
│  │  ┌──────────────┐                  ┌──────────────┐       │  │
│  │  │ VTK MODULE   │                  │  PCL MODULE  │       │  │
│  │  │ (Current)    │                  │  (Future)    │       │  │
│  │  │              │                  │              │       │  │
│  │  │ • Surface    │                  │ • Organized  │       │  │
│  │  │   rendering  │                  │   clouds     │       │  │
│  │  │ • Color map  │                  │ • PCA        │       │  │
│  │  │ • Interactive│                  │ • Outlier    │       │  │
│  │  │   3D view    │                  │   detection  │       │  │
│  │  └──────────────┘                  └──────────────┘       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
                    ┌─────────────────┐
                    │   CSV Input     │
                    │  (comma-sep)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────────────────┐
                    │  STEP 1: Load CSV               │
                    │  • Parse file                   │
                    │  • Create Eigen::MatrixXf       │
                    │  • Store all data points        │
                    └────────┬────────────────────────┘
                             │
                    ┌────────▼────────────────────────┐
                    │  STEP 2: Handle Invalid Points  │
                    │  • Detect -99999.9999           │
                    │  • Replace with NaN             │
                    │  • Count valid/invalid          │
                    └────────┬────────────────────────┘
                             │
                    ┌────────▼────────────────────────┐
                    │  STEP 3: Apply Filtering        │
                    │                                 │
      ┌─────────────┤  CURRENT: Pass-through          │
      │             │  FUTURE:  PCA outlier removal   │◄─── PCL
      │             │                                 │
      │             └────────┬────────────────────────┘
      │                      │
      │             ┌────────▼────────────────────────┐
      │             │  STEP 4: Downsample             │
      │             │  • Sample rows/cols linearly    │
      │             │  • Create display subset        │
      │             │  • Maintain index mapping       │
      │             └────────┬────────────────────────┘
      │                      │
      │             ┌────────▼────────────────────────┐
      │             │  STEP 5: Visualize              │
      │             │  • Convert to VTK grid          │
      │             │  • Apply color mapping          │
      │             │  • Render 3D surface            │
      │             └────────┬────────────────────────┘
      │                      │
      │                      ▼
      │             ┌─────────────────┐
      │             │ Interactive 3D  │
      │             │  Visualization  │
      │             └─────────────────┘
      │
      │  [FUTURE PCL FILTERING DETAIL]
      │
      └───────►┌────────────────────────────────────┐
               │  Tile-Based PCA Processing         │
               │                                    │
               │  For each tile:                    │
               │    1. Eigen → PCL (organized)      │
               │    2. Compute local PCA            │
               │    3. Find best-fit plane          │
               │    4. Calculate distances          │
               │    5. Compute MAD                  │
               │    6. Detect outliers              │
               │    7. Mark as NaN                  │
               │                                    │
               │  Return: Cleaned Eigen matrix      │
               └────────────────────────────────────┘
```

## Library Responsibilities

```
┌──────────────────────────────────────────────────────────────┐
│                       EIGEN                                  │
│  Role: Numerical data storage and operations                 │
│                                                              │
│  • Matrix storage (raw_data, filtered_data, display_data)    │
│  • Element-wise operations                                   │
│  • Efficient memory layout                                   │
│  • NumPy-like interface                                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                       VTK                                    │
│  Role: 3D visualization of structured grids                  │
│                                                              │
│  • Structured grid representation                            │
│  • Surface rendering                                         │
│  • Color mapping (height → color)                            │
│  • Interactive controls                                      │
│  • Camera manipulation                                       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                       PCL (Future)                           │
│  Role: Geometric analysis and outlier detection              │
│                                                              │
│  • Organized point cloud support                             │
│  • PCA computation                                           │
│  • Covariance analysis                                       │
│  • Plane fitting                                             │
│  • Neighbor search (KdTree)                                  │
└──────────────────────────────────────────────────────────────┘
```

## Class Structure

```
namespace keyence {

    ┌─────────────────────────────────────────────┐
    │         struct AnalysisResult               │
    ├─────────────────────────────────────────────┤
    │  + string filename                          │
    │  + string status                            │
    │  + int rows, cols                           │
    │  + int valid_points, invalid_points         │
    │  + Eigen::MatrixXf raw_data                 │
    │  + Eigen::MatrixXf filtered_data            │
    │  + Eigen::MatrixXf display_data             │
    └─────────────────────────────────────────────┘
                      ▲
                      │ returns
                      │
    ┌─────────────────┴────────────────────────────┐
    │         class KeyenceAnalyzer                │
    ├──────────────────────────────────────────────┤
    │  - float invalid_value_                      │
    │  - float pixel_pitch_x_, pixel_pitch_y_      │
    │  - map<string, AnalysisResult> results_      │
    ├──────────────────────────────────────────────┤
    │  PUBLIC:                                     │
    │  + analyzeSingleFile(filepath, sample_size)  │
    │  + analyzeAllFiles(folder, sample_size)      │
    │                                              │
    │  PRIVATE PIPELINE:                           │
    │  • loadCSV(filepath, data)                   │
    │  • replaceInvalidValues(data)                │
    │  • applyFiltering(data) ◄──┐                 │
    │  • createDisplaySample(data)│                │
    │  • visualizeSurface(data)   │                │
    └─────────────────────────────┼────────────────┘
                                  │
                [FUTURE]          │
                                  │
    ┌─────────────────────────────┴────────────────┐
    │      PCL Filtering Module (Future)           │
    ├──────────────────────────────────────────────┤
    │  + filterOutliersWithPCA()                   │
    │  + eigenToPCLOrganized()                     │
    │  + computeTilePCA()                          │
    │  + detectOutliersMAD()                       │
    │  + computeMAD()                              │
    └──────────────────────────────────────────────┘
}
```

## Memory Layout

```
┌────────────────────────────────────────────────────────────┐
│                    IN-MEMORY STRUCTURE                     │
└────────────────────────────────────────────────────────────┘

Raw Data Matrix (Full Resolution)
┌────────────────────────────────────┐
│  Eigen::MatrixXf raw_data          │
│  Size: rows × cols                 │
│  Type: float (4 bytes/point)       │
│  Content: Height in mm             │
│  Invalid: NaN                      │
│                                    │
│  Example: 3000×3000 = 36 MB        │
└────────────────────────────────────┘
            │
            │ Copy (with future filtering)
            ▼
┌────────────────────────────────────┐
│  Eigen::MatrixXf filtered_data     │
│  Size: rows × cols                 │
│  Content: Outliers → NaN           │
│                                    │
│  Example: 3000×3000 = 36 MB        │
└────────────────────────────────────┘
            │
            │ Downsample
            ▼
┌────────────────────────────────────┐
│  Eigen::MatrixXf display_data      │
│  Size: sample × sample             │
│  Content: Linearly sampled         │
│                                    │
│  Example: 500×500 = 1 MB           │
└────────────────────────────────────┘
            │
            │ Convert
            ▼
┌────────────────────────────────────┐
│  vtkStructuredGrid                 │
│  Contains: Points + Scalars        │
│  Used for: VTK rendering only      │
└────────────────────────────────────┘
```

## Future PCL Integration Detail

```
┌───────────────────────────────────────────────────────────┐
│            PCL-BASED OUTLIER FILTERING                    │
└───────────────────────────────────────────────────────────┘

Step 1: Conversion
┌──────────────────┐         ┌────────────────────┐
│ Eigen::MatrixXf  │ ──────► │ pcl::PointCloud    │
│  Height matrix   │         │  (organized)       │
│  rows × cols     │         │  width × height    │
└──────────────────┘         └────────────────────┘
                                      │
Step 2: Tiling                        │
                                      ▼
                    ┌──────────────────────────────┐
                    │   Divide into tiles          │
                    │   (e.g., 1000×1000 each)     │
                    └──────────────────────────────┘
                                      │
Step 3: Per-Tile Processing           │
                                      ▼
            ┌─────────────────────────────────────┐
            │  For each tile:                     │
            │                                     │
            │  1. Extract tile points             │
            │  2. Compute PCA:                    │
            │     • Covariance matrix             │
            │     • Eigenvalues, eigenvectors     │
            │     • Best-fit plane                │
            │  3. Calculate distances to plane    │
            │  4. Compute MAD (robust statistic)  │
            │  5. Threshold: |dist|/MAD > 10      │
            │  6. Mark outliers → NaN             │
            └─────────────────────────────────────┘
                                      │
Step 4: Conversion Back               │
                                      ▼
┌────────────────────┐       ┌──────────────────┐
│  Eigen::MatrixXf   │ ◄──── │  PCL Results     │
│  Cleaned data      │       │  Outlier mask    │
│  Outliers = NaN    │       └──────────────────┘
└────────────────────┘
```

## File Organization

```
keyence_analyzer/
│
├── Core Components (CURRENT) ✅
│   ├── keyence_analyzer.h        Main class definition
│   ├── keyence_analyzer.cpp      Implementation
│   └── main.cpp                  CLI interface
│
├── Future Components (PCL) ⧗
│   ├── keyence_filtering.h       Filtering interface
│   ├── keyence_filtering.cpp     PCA outlier detection
│   └── conversion_utils.h/cpp    Eigen ↔ PCL conversion
│
├── Build System
│   └── CMakeLists.txt            Build configuration
│
└── Documentation
    ├── README.md                 Full documentation
    ├── QUICKSTART.md             Quick start guide
    └── ARCHITECTURE.md           This file
```

