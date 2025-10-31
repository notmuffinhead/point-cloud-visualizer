# Batch Processing Guide: Process Entire Folders

## 🚀 Quick Start (3 Steps)

### Step 1: Put your CSV files in a folder

```
KEYENCE_DATASET/
├── file1.csv
├── file2.csv
├── file3.csv
└── ...
```

### Step 2: Create a simple script

```python
from keyence_batch import quick_keyence_analysis_and_clean

# Process ALL CSV files in folder
analyzer, results = quick_keyence_analysis_and_clean(
    folder_path="KEYENCE_DATASET",  # Your folder path here!
    subsample=4,                     # Fast processing
    curvature_threshold=0.1,         # Spike removal threshold
    n_jobs=-1                        # Use all CPU cores
)
```

### Step 3: Run it!

```bash
python your_script.py
```

**That's it!** All files will be:
- Analyzed for surface properties
- Cleaned of spikes
- Saved to `cleaned_output/` folder as `.ply` files

---

## 📂 What You Get

### Input:
```
KEYENCE_DATASET/
├── scan001.csv  (51M points with spikes)
├── scan002.csv  (45M points with spikes)
└── scan003.csv  (38M points with spikes)
```

### Output:
```
cleaned_output/
├── scan001_cleaned.ply  (48M points, spikes removed!)
├── scan002_cleaned.ply  (43M points, spikes removed!)
└── scan003_cleaned.ply  (36M points, spikes removed!)
```

Plus console output showing:
- Progress for each file
- Before/after statistics
- Processing time
- Summary report

---

## 🎯 Complete Example

```python
from keyence_batch import quick_keyence_analysis_and_clean

# Process entire folder
analyzer, results = quick_keyence_analysis_and_clean(
    folder_path="C:/Users/maggi/OneDrive/Documents/HELP/files",  # Your path
    subsample=4,              # Process 1/16th of points (still detailed!)
    curvature_threshold=0.1,  # Remove spikes with curvature > 0.1
    k_neighbors=20,           # Use 20 neighbors for analysis
    n_jobs=-1,                # Use all CPU cores
    visualize=False,          # Don't show plots (faster for batch)
    output_dir='cleaned_output'  # Where to save results
)

# Print summary
print(f"\nProcessed {len(results)} files")
for filename, stats in results.items():
    if stats['status'] == 'success':
        removed_pct = stats['percent_removed']
        print(f"  ✅ {filename}: {removed_pct:.1f}% spikes removed")
    else:
        print(f"  ❌ {filename}: Failed")
```

---

## ⚙️ Configuration Options

### Fast Processing (Recommended for 51M points)
```python
analyzer, results = quick_keyence_analysis_and_clean(
    'KEYENCE_DATASET',
    subsample=4,           # ~13M points per file, ~5 min each
    curvature_threshold=0.1,
    visualize=False        # Skip plots for speed
)
```

### High Quality Processing
```python
analyzer, results = quick_keyence_analysis_and_clean(
    'KEYENCE_DATASET',
    subsample=2,           # ~25M points per file, ~10 min each
    curvature_threshold=0.05,  # More aggressive filtering
    k_neighbors=30         # More accurate analysis
)
```

### Very Fast Preview
```python
analyzer, results = quick_keyence_analysis_and_clean(
    'KEYENCE_DATASET',
    subsample=8,           # ~6M points per file, ~3 min each
    curvature_threshold=0.1
)
```

---

## 🔧 Different Filtering Methods

### Method 1: Curvature-Based (Default, Recommended)
```python
analyzer, results = quick_keyence_analysis_and_clean(
    'KEYENCE_DATASET',
    filter_method='curvature',
    curvature_threshold=0.1  # Adjust this based on your data
)
```

### Method 2: Statistical Outlier Removal
```python
from keyence_batch import KeyenceAnalyzerComplete

analyzer = KeyenceAnalyzerComplete(n_jobs=-1)
results = analyzer.analyze_all_files(
    folder_path='KEYENCE_DATASET',
    subsample=4,
    filter_method='statistical',
    filter_params={'k_neighbors': 20, 'std_ratio': 2.0}
)
```

### Method 3: Multi-Stage (Most Aggressive)
```python
analyzer, results = quick_keyence_analysis_and_clean(
    'KEYENCE_DATASET',
    filter_method='multi'  # Runs multiple filters
)
```

---

## 📊 Understanding the Output

### Console Output Example:
```
🚀 BATCH ANALYSIS + SPIKE REMOVAL
📂 Folder: KEYENCE_DATASET
📄 Found 3 CSV files
🔧 Filter method: curvature
💾 Output dir: cleaned_output
======================================================================

======================================================================
PROCESSING FILE 1 OF 3
======================================================================

📁 Loading: scan001.csv
📋 Shape: 4096 × 12800
📊 Valid: 51,003,456/52,428,800 (97.3%)

🚀 OPTIMIZED computation
   Method: Multiprocessing + Numba JIT
   Jobs: all cores

🔬 Step 1/3: Computing normals...
   ✅ Done in 45.2s

🔬 Step 2/3: Finding neighbors (parallel)...
   ✅ Done in 89.4s

🔬 Step 3/3: Computing planarity & curvature (Numba)...
   ✅ Done in 52.1s

✅ TOTAL TIME: 186.7s (68,274 pts/sec)

📊 ORIGINAL DATA STATISTICS:
   Points: 12,750,864
   Curvature: mean=0.015432, max=0.456789
   Planarity: mean=0.8234, min=0.0012

======================================================================
🔧 SPIKE REMOVAL
======================================================================

🔍 Curvature-Based Filtering (threshold=0.1)
   Original points: 12,750,864
   Curvature range: [0.000001, 0.456789]
   Curvature mean: 0.015432
   Filtered points: 12,489,372
   ❌ Removed: 261,492 (2.05%)
   ⏱️  Time: 1.34s

📊 FILTERED DATA STATISTICS:
   Points: 12,489,372
   Curvature: mean=0.012234, max=0.099876
   Planarity: mean=0.8567, min=0.0012

💾 Saved cleaned point cloud: cleaned_output/scan001_cleaned.ply

🎉 ANALYSIS COMPLETE!
======================================================================

⏸️  Press Enter to continue to next file (2/3)...
```

---

## 💾 Working with Output Files

### Load Cleaned Point Cloud

```python
import open3d as o3d

# Load cleaned data
pcd = o3d.io.read_point_cloud("cleaned_output/scan001_cleaned.ply")
points = np.asarray(pcd.points)

print(f"Loaded {len(points):,} clean points!")

# Visualize
o3d.visualization.draw_geometries([pcd])
```

### Convert to Other Formats

```python
# PLY to XYZ
o3d.io.write_point_cloud("output.xyz", pcd, write_ascii=True)

# PLY to PCD
o3d.io.write_point_cloud("output.pcd", pcd)

# Extract points to NumPy
import numpy as np
points = np.asarray(pcd.points)
np.save("points.npy", points)
```

---

## 📈 Expected Performance

### For Multiple 51M Point Files:

| Configuration | Time per File | Quality |
|--------------|---------------|---------|
| subsample=1 | ~25 min | Maximum |
| subsample=2 | ~10 min | High |
| subsample=4 | ~5 min | Good (recommended) |
| subsample=8 | ~3 min | Preview |

**Example**: 10 files with 51M points each
- subsample=4: ~50 minutes total
- subsample=8: ~30 minutes total

---

## 🎯 Advanced Features

### Custom Configuration Per File

```python
from keyence_batch import KeyenceAnalyzerComplete

analyzer = KeyenceAnalyzerComplete(n_jobs=-1)

# Process each file with different settings
files = ['file1.csv', 'file2.csv', 'file3.csv']
thresholds = [0.1, 0.08, 0.12]  # Different threshold per file

for filepath, threshold in zip(files, thresholds):
    result = analyzer.analyze_and_filter(
        filepath,
        subsample=4,
        filter_method='curvature',
        filter_params={'threshold': threshold},
        save_output=True,
        output_dir='cleaned_output'
    )
    print(f"Processed {filepath} with threshold {threshold}")
```

### Disable Auto-Continue Prompt

If you don't want to press Enter between files, modify the code:

In `keyence_batch.py`, comment out this line (around line 377):
```python
# if i < len(csv_files):
#     input(f"\n⏸️  Press Enter to continue to next file ({i+1}/{len(csv_files)})...")
```

---

## ❓ Troubleshooting

### "No CSV files found"
→ Check your folder path is correct
```python
# Use absolute path on Windows
folder_path = r"C:\Users\maggi\OneDrive\Documents\HELP\files"
```

### "Out of memory"
→ Increase subsample
```python
subsample=8  # or even 16 for very large files
```

### "Takes too long"
→ Use higher subsample for initial testing
```python
subsample=8  # Fast preview first
# Then once you're happy with settings:
subsample=4  # For final processing
```

### "Too many spikes removed"
→ Increase threshold
```python
curvature_threshold=0.15  # or 0.2
```

### "Spikes still remain"
→ Decrease threshold or use multi-stage
```python
curvature_threshold=0.05  # More aggressive
# or
filter_method='multi'  # Multiple filters
```

---

## 📋 Checklist for Batch Processing

- [ ] Install requirements: `pip install joblib numba`
- [ ] Put all CSV files in one folder
- [ ] Create run script with correct folder path
- [ ] Test on one file first (set visualize=True)
- [ ] Adjust curvature_threshold based on histogram
- [ ] Run batch processing on all files
- [ ] Check output files in `cleaned_output/` folder
- [ ] Verify spike removal worked
- [ ] Save/backup cleaned files

---

## 🚀 Ready to Process!

```python
from keyence_batch import quick_keyence_analysis_and_clean

# Process your entire dataset!
analyzer, results = quick_keyence_analysis_and_clean(
    folder_path="YOUR_FOLDER_PATH_HERE",
    subsample=4,
    curvature_threshold=0.1,
    n_jobs=-1,
    output_dir='cleaned_output'
)

print("🎉 All files processed and cleaned!")
```

---

**Questions? Check the other documentation files or run with `visualize=True` on a single file first to understand the filtering behavior.**
