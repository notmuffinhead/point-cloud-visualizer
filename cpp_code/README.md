# Keyence Profilometer Data Analyzer (C++)

A C++ implementation for processing and visualizing 3D surface topography data from Keyence profilometer CSV files, with a modular architecture designed for future PCL-based outlier filtering.

## Overview

This analyzer implements a clean pipeline architecture:
```
Load CSV → Handle Invalid Points → Filter (future: PCA) → Sample → Visualize
```

Each stage is independent and can be enhanced without affecting other parts of the pipeline.

## Features

### Current Implementation ✅
- **Full dataset loading**: Loads all data points from CSV files into Eigen matrices
- **Smart sampling**: Adjustable display sampling for performance (default 500×500)
- **Physical coordinates**: Accurate 2.5 μm pixel pitch in X and Y directions
- **Invalid data handling**: Automatically converts -99999.9999 markers to NaN
- **3D visualization**: Interactive VTK-based surface plots with color mapping
- **Batch processing**: Analyze single files or entire folders
- **Memory efficient**: Loads full data but samples strategically for display

### Future Enhancements ⧗
- **PCL-based outlier filtering**: Tile-based PCA analysis with MAD (Median Absolute Deviation)
- **Statistical analysis**: Surface roughness, peak detection, curvature analysis
- **Export capabilities**: Save cleaned data to various formats

## Architecture

### Data Flow

```
┌─────────────────┐
│   CSV File      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Eigen::MatrixXf │  ◄── Raw data (full resolution)
│   (raw_data)    │
└────────┬────────┘
         │
┌────────▼───────────────────────┐
│  Replace invalid (-99999.9999) │
│         with NaN               │
└────────┬───────────────────────┘
         │
┌────────▼────────────────────────┐
│  Apply Filtering                │
│  Current: Pass-through          │
│  Future: PCA outlier detection  │
└────────┬────────────────────────┘
         │
┌────────▼────────┐
│ Eigen::MatrixXf │  ◄── Filtered data
│ (filtered_data) │
└────────┬────────┘
         │
┌────────▼────────┐
│  Downsample     │
│  (for display)  │
└────────┬────────┘
         │
┌────────▼────────┐
│ Eigen::MatrixXf │  ◄── Display subset
│ (display_data)  │
└────────┬────────┘
         │
┌────────▼──────────┐
│   Convert to      │
│ VTK StructuredGrid│
└────────┬──────────┘
         │
┌────────▼────────┐
│  3D Surface     │
│  Visualization  │
└─────────────────┘
```

### Class Structure

```cpp
namespace keyence {
    
    struct AnalysisResult {
        Eigen::MatrixXf raw_data;      // Direct from CSV
        Eigen::MatrixXf filtered_data; // After filtering
        Eigen::MatrixXf display_data;  // Sampled for viz
        // ... metadata
    };
    
    class KeyenceAnalyzer {
        // STEP 1: Data Loading
        bool loadCSV(filepath, data);
        
        // STEP 2: Invalid Point Handling
        int replaceInvalidValues(data);
        
        // STEP 3: Filtering (FUTURE: PCL Integration Point)
        Eigen::MatrixXf applyFiltering(raw_data);
        // [FUTURE] filterOutliersWithPCA(data, tile_size, mad_threshold);
        
        // STEP 4: Sampling
        Eigen::MatrixXf createDisplaySample(full_data, sample_size);
        
        // STEP 5: Visualization (VTK)
        void visualizeSurface(heights, filename);
    };
}
```

## Keyence Specifications

- **X pixel pitch**: 2.5 μm
- **Y pixel pitch**: 2.5 μm (nominal, may be slightly distorted)
- **Z reference**: Center of sensor FOV
- **X/Y origin**: Top left corner of image
- **Invalid data marker**: -99999.9999 (converted to NaN)

## Dependencies

### Required

1. **Eigen3** (3.3+): Linear algebra library
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libeigen3-dev
   
   # macOS
   brew install eigen
   ```

2. **VTK** (8.0+): Visualization Toolkit
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libvtk9-dev
   
   # macOS
   brew install vtk
   ```

3. **C++17 compiler**: GCC 7+, Clang 5+, or MSVC 2017+
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential cmake
   
   # macOS
   xcode-select --install
   brew install cmake
   ```

### Optional

- **OpenMP**: For parallel processing (usually included with GCC)
- **PCL** (1.8+): Point Cloud Library (for future filtering features)

## Building

### Quick Start

```bash
# Clone or extract the source code
cd keyence_analyzer

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..

# The executable will be in build/keyence_analyzer
```

### Build Options

```bash
# Release build (optimized, recommended)
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Debug build (for development)
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Specify custom library paths
cmake -DVTK_DIR=/path/to/vtk/lib/cmake/vtk-9.0 \
      -DEigen3_DIR=/path/to/eigen3/share/eigen3/cmake \
      ..
make

# Install to system
sudo make install
```

### Build Output

```
build/
├── keyence_analyzer        # Main executable
└── ...                     # CMake files
```

## Usage

### Command Line

```bash
# Analyze a single CSV file
./keyence_analyzer data.csv

# Analyze with custom display sampling (higher = more detail)
./keyence_analyzer data.csv 1000

# Analyze all CSV files in a folder
./keyence_analyzer KEYENCE_DATASET

# Analyze folder with custom sampling
./keyence_analyzer KEYENCE_DATASET 800
```

### Display Sample Size Guidelines

| Size | Points | Use Case | Performance |
|------|--------|----------|-------------|
| 200  | 40K    | Quick preview | Very fast |
| 500  | 250K   | Default, good balance | Fast |
| 800  | 640K   | Detailed view | Medium |
| 1000 | 1M     | Maximum detail | Slower |

### Programmatic Usage

```cpp
#include "keyence_analyzer.h"

int main() {
    // Method 1: Single file analysis
    keyence::KeyenceAnalyzer analyzer;
    auto result = analyzer.analyzeSingleFile("data.csv", 500);
    
    std::cout << "Valid points: " << result.valid_points << std::endl;
    
    // Access the data matrices
    // result.raw_data - complete Eigen::MatrixXf (all points)
    // result.display_data - downsampled for visualization
    
    // Method 2: Batch processing
    auto results = analyzer.analyzeAllFiles("KEYENCE_DATASET", 500);
    
    for (const auto& [filename, result] : results) {
        std::cout << "File: " << filename 
                  << " - Status: " << result.status << std::endl;
    }
    
    // Method 3: Convenience function (auto-detect file/folder)
    keyence::quickKeyenceAnalysis("path_to_data");
    
    return 0;
}
```

## Data Format

### Input CSV Format
```
0.123,0.125,0.128,...
0.134,0.132,0.130,...
-99999.9999,0.145,0.142,...  # Invalid point
...
```

- Values in millimeters
- Comma-separated
- No header row
- Invalid points marked as -99999.9999

### Output Data Structure

```cpp
struct AnalysisResult {
    std::string filename;
    std::string status;              // "success" or "failed"
    int rows, cols;                  // Matrix dimensions
    int valid_points, invalid_points;
    Eigen::MatrixXf raw_data;        // Original CSV data
    Eigen::MatrixXf filtered_data;   // After filtering (future)
    Eigen::MatrixXf display_data;    // Downsampled for viz
};
```

## Visualization Controls

The interactive 3D visualization window supports:

| Control | Action |
|---------|--------|
| **Left mouse drag** | Rotate view |
| **Right mouse drag** | Zoom in/out |
| **Middle mouse drag** | Pan view |
| **Mouse wheel** | Zoom |
| **'r' key** | Reset camera |
| **'s' key** | Toggle surface/wireframe |
| **'q' key** | Quit window |


## Performance Notes

### Memory Usage

- **Full data**: rows × cols × 4 bytes (float)
  - Example: 3000×3000 = 36 MB per file
- **Display data**: sample_size² × 4 bytes
  - Example: 500×500 = 1 MB per visualization

### Processing Speed

| Operation | Time (3000×3000) |
|-----------|------------------|
| CSV loading | 1-2 seconds |
| Invalid replacement | < 0.1 seconds |
| Sampling | < 0.1 seconds |
| VTK rendering | Real-time |

### Optimization Tips

1. **Adjust display sample size** based on needs
   - Preview: 200-300
   - Standard: 500
   - Detailed: 800-1000

2. **Close visualization windows** to speed up batch processing

3. **Use Release build** for production (5-10× faster than Debug)

4. **Enable OpenMP** for future parallel processing

## Design Principles

### 1. Separation of Concerns
```
Data Storage:  Eigen (numerical operations)
Analysis:      PCL   (geometric algorithms) [future]
Visualization: VTK   (surface rendering)
```

### 2. Pipeline Architecture
```
Load → Filter → Sample → Visualize
       ↑
       [PCL integration point]
```

### 3. Organized Structure Preservation
- Maintain row/column indexing throughout
- Preserve spatial relationships
- Easy conversion between representations

### 4. Optional Features
- Filtering can be enabled/disabled
- Doesn't affect rest of pipeline
- Backward compatible


