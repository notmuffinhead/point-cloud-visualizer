# Quick Start Guide - Keyence Analyzer C++

## Installation & Build (5 minutes)

### Step 1: Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev libvtk9-dev

# macOS
brew install cmake eigen vtk
```

### Step 2: Build

```bash
# Navigate to project directory
cd keyence_analyzer

# Create build directory
mkdir build && cd build

# Build
cmake --build . --config Release

# You now have: build/keyence_analyzer
```

### Step 3: Run

```bash
# Single file
./keyence_analyzer.exe /path/to/data.csv

# With custom sampling (more detail)
./keyence_analyzer.exe /path/to/data.csv 1000

# Entire folder
./keyence_analyzer.exe /path/to/KEYENCE_DATASET

# Folder with custom sampling
./keyence_analyzer.exe /path/to/KEYENCE_DATASET 800
```

## Usage Examples

### Basic Analysis
```bash
# Analyze one file with default settings (500x500 display)
./keyence_analyzer.exe surface_scan.csv
```

**Output:**
- Console shows: file info, point counts, physical dimensions
- Interactive 3D window opens
- Use mouse to rotate, zoom, pan
- Close window to continue

### High-Detail Visualization
```bash
# Use 1000x1000 display for maximum detail
./keyence_analyzer.exe surface_scan.csv 1000
```

**Use when:**
- Need to see fine features
- File is not too large (< 5000×5000)
- Have good GPU

### Batch Processing
```bash
# Analyze all CSV files in folder
./keyence_analyzer.exe ./KEYENCE_DATASET
```

**Behavior:**
- Processes files alphabetically
- Shows each visualization
- Press Enter to continue to next file
- Summary at end

### Quick Preview
```bash
# Fast preview with 200x200 display
./keyence_analyzer.exe large_file.csv 200
```

**Use when:**
- Quick quality check
- Very large files
- Slow computer

## Programmatic Usage

### Example 1: Single File Analysis

```cpp
#include "keyence_analyzer.h"

int main() {
    // Create analyzer
    keyence::KeyenceAnalyzer analyzer;
    
    // Analyze file
    auto result = analyzer.analyzeSingleFile("data.csv", 500);
    
    // Check status
    if (result.status == "success") {
        std::cout << "Analyzed " << result.valid_points << " points\n";
        std::cout << "Dimensions: " << result.rows << "×" << result.cols << "\n";
        
        // Access data matrices
        // result.raw_data - Eigen::MatrixXf with all points
        // result.filtered_data - After filtering (currently same as raw)
        // result.display_data - Downsampled version
        
        // Example: Calculate height range
        float min_height = result.raw_data.minCoeff();
        float max_height = result.raw_data.maxCoeff();
        std::cout << "Height range: " << min_height << " to " 
                  << max_height << " mm\n";
    }
    
    return 0;
}
```

### Example 2: Batch Processing

```cpp
#include "keyence_analyzer.h"

int main() {
    keyence::KeyenceAnalyzer analyzer;
    
    // Analyze all files in folder
    auto results = analyzer.analyzeAllFiles("KEYENCE_DATASET", 500);
    
    // Process results
    for (const auto& [filename, result] : results) {
        if (result.status == "success") {
            std::cout << filename << ": " 
                      << result.valid_points << " valid points\n";
        } else {
            std::cout << filename << ": FAILED\n";
        }
    }
    
    return 0;
}
```

### Example 3: Custom Analysis

```cpp
#include "keyence_analyzer.h"

int main() {
    keyence::KeyenceAnalyzer analyzer;
    auto result = analyzer.analyzeSingleFile("data.csv", 500);
    
    // Custom analysis on the full data
    const auto& data = result.raw_data;
    
    // Calculate statistics (excluding NaN)
    float sum = 0;
    int count = 0;
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            if (!std::isnan(data(i, j))) {
                sum += data(i, j);
                count++;
            }
        }
    }
    
    float mean_height = sum / count;
    std::cout << "Mean height: " << mean_height << " mm\n";
    
    return 0;
}
```

## Visualization Controls

| Action | Control |
|--------|---------|
| Rotate | Left mouse drag |
| Zoom | Right mouse drag OR scroll wheel |
| Pan | Middle mouse drag |
| Reset view | Press 'r' |
| Wireframe | Press 's' |
| Quit | Press 'q' OR close window |

## File Structure

```
keyence_analyzer/
├── keyence_analyzer.h       ← Main class definition
├── keyence_analyzer.cpp     ← Implementation
├── keyence_filtering.h      ← [FUTURE] PCL filtering interface
├── keyence_filtering.cpp    ← [FUTURE] PCL filtering implementation
├── main.cpp                 ← Command-line interface
├── CMakeLists.txt           ← Build configuration
└── README.md                ← Full documentation
```
