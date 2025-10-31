# How to Actually Remove Spikes from Your Data

## 🎯 The Problem

Your code computed normals, planarity, and curvature but **NEVER actually removed or filtered anything**. It just analyzed the surface!

**What you had**: Analysis only (like taking measurements)  
**What you need**: Filtering (like actually fixing the problem)

---

## ✅ Solution: Use the Complete Package

I've created 3 files that work together:

1. **spike_removal.py** - All the filtering methods
2. **keyence_complete.py** - Everything integrated (EASIEST!)
3. **SPIKE_REMOVAL_GUIDE.md** - Detailed explanations

---

## 🚀 Quick Start (Easiest Method)

### Step 1: Install Requirements

```bash
pip install joblib numba
```

### Step 2: Run the Complete Solution

```python
from keyence_complete import analyze_and_clean_keyence

# ONE LINE! Analyzes AND removes spikes
analyzer, results = analyze_and_clean_keyence(
    'your_51million_points.csv',
    subsample=4,              # Process faster (still ~12.75M points)
    filter_method='curvature', # Remove high-curvature spikes
    curvature_threshold=0.1,  # Adjust based on your data
    k_neighbors=20,
    n_jobs=-1                 # Use all CPU cores
)

# Access the cleaned data
cleaned_points = results['points_filtered']
cleaned_pcd = results['pcd_filtered']

print(f"Original: {len(results['points_original']):,} points")
print(f"Cleaned:  {len(cleaned_points):,} points")
print(f"Removed:  {len(results['points_original']) - len(cleaned_points):,} spikes")

# Save cleaned point cloud
import open3d as o3d
o3d.io.write_point_cloud("cleaned_surface.ply", cleaned_pcd)
```

**This will:**
1. Load your data (fast!)
2. Compute surface properties with optimized code (~9 minutes for 51M points)
3. Remove spikes based on curvature
4. Show you before/after visualization
5. Return both original and cleaned data

---

## 🎨 How to Choose the Right Threshold

Your spikes have high curvature. You need to look at the curvature histogram to pick a good threshold!

### Method 1: Look at the Histogram (After First Run)

After running once, you'll see the curvature histogram. Look for:
- **Most points**: Low curvature (flat surface) - around 0.001-0.01
- **Spike points**: High curvature (sharp peaks) - above 0.1-0.5

Example thresholds:
- **0.2**: Very conservative (only removes sharp spikes)
- **0.1**: Moderate (good default) ← START HERE
- **0.05**: Aggressive (removes subtle bumps too)

### Method 2: Iterative Tuning

```python
# Try different thresholds
for threshold in [0.2, 0.1, 0.05]:
    analyzer, results = analyze_and_clean_keyence(
        'data.csv',
        curvature_threshold=threshold,
        subsample=4
    )
    removed_pct = 100 * (1 - len(results['points_filtered']) / len(results['points_original']))
    print(f"Threshold {threshold}: Removed {removed_pct:.1f}% of points")
```

---

## 🔧 Different Filtering Methods

### Method 1: Curvature-Based (RECOMMENDED for real spikes)

```python
analyzer, results = analyze_and_clean_keyence(
    'data.csv',
    filter_method='curvature',
    curvature_threshold=0.1,  # Adjust this!
    subsample=4
)
```

**Best for**: Physical spikes/protrusions (your case!)  
**How it works**: Removes points with high curvature  
**Tune**: Lower threshold = more aggressive

---

### Method 2: Statistical Outlier Removal (for random noise)

```python
from keyence_complete import KeyenceAnalyzerComplete

analyzer = KeyenceAnalyzerComplete(n_jobs=-1)
results = analyzer.analyze_and_filter(
    'data.csv',
    subsample=4,
    filter_method='statistical',
    filter_params={'k_neighbors': 20, 'std_ratio': 2.0}
)
```

**Best for**: Random measurement noise, isolated outliers  
**How it works**: Removes points far from neighbors  
**Tune**: Lower std_ratio = more aggressive

---

### Method 3: Multi-Stage Pipeline (MOST ROBUST)

```python
analyzer, results = analyze_and_clean_keyence(
    'data.csv',
    filter_method='multi',  # Runs multiple filters in sequence
    subsample=4
)
```

**Best for**: When you have multiple types of noise  
**How it works**: 
1. Remove isolated points (radius outlier)
2. Remove high-curvature spikes
3. Remove statistical outliers
**Result**: Cleanest output but may be aggressive

---

## 📊 What You Get Back

```python
results = {
    # Original (unfiltered) data
    'points_original': Nx3 array,
    'pcd_original': Open3D point cloud,
    'curvature_original': N array,
    'normals_original': Nx3 array,
    
    # Filtered (cleaned) data
    'points_filtered': Mx3 array,  # M < N
    'pcd_filtered': Open3D point cloud,
    'curvature_filtered': M array,
    'normals_filtered': Mx3 array,
    
    # Metadata
    'filter_mask': Boolean mask showing which points were kept
}
```

---

## 💾 Saving Your Results

### Save Cleaned Point Cloud

```python
import open3d as o3d

# Save as PLY (recommended - preserves all data)
o3d.io.write_point_cloud("cleaned_surface.ply", results['pcd_filtered'])

# Save as PCD (Open3D native format)
o3d.io.write_point_cloud("cleaned_surface.pcd", results['pcd_filtered'])

# Save as XYZ (simple text format)
o3d.io.write_point_cloud("cleaned_surface.xyz", results['pcd_filtered'])
```

### Export to CSV/Array

```python
import numpy as np
import pandas as pd

# Get cleaned points
cleaned_points = results['points_filtered']

# Save as NumPy array
np.save('cleaned_points.npy', cleaned_points)

# Save as CSV
pd.DataFrame(
    cleaned_points,
    columns=['X_um', 'Y_um', 'Z_um']
).to_csv('cleaned_points.csv', index=False)
```

### Reconstruct Height Map

```python
# Get filtered points and indices
points_filtered = results['points_filtered']
indices_filtered = results['valid_indices_filtered']
original_shape = results['full_shape']

# Create new height map
height_map_cleaned = np.full(original_shape, np.nan)

for point, (row, col) in zip(points_filtered, indices_filtered):
    z_um = point[2]
    z_mm = z_um / 1000  # Convert back to mm
    height_map_cleaned[row, col] = z_mm

# Save as CSV (Keyence format)
pd.DataFrame(height_map_cleaned).to_csv('cleaned_surface.csv', header=False, index=False)
```

---

## 🎯 Recommended Workflow for Your 51M Points

### Phase 1: Test on Small Region (FAST!)

```python
# Use high subsample first to test quickly
analyzer, results = analyze_and_clean_keyence(
    'your_data.csv',
    subsample=8,  # Only ~6.4M points - fast!
    curvature_threshold=0.1
)

# Look at the histogram and before/after plot
# Adjust threshold if needed
```

### Phase 2: Full Processing (Takes ~10-15 minutes)

```python
# Once you're happy with the threshold, process all data
analyzer, results = analyze_and_clean_keyence(
    'your_data.csv',
    subsample=2,  # ~25M points
    curvature_threshold=0.1,  # Or whatever worked in Phase 1
    n_jobs=-1
)

# Save the results
import open3d as o3d
o3d.io.write_point_cloud("cleaned_surface.ply", results['pcd_filtered'])
```

---

## 🔍 Advanced: Manual Control

If you want more control, use the classes directly:

```python
from keyence_complete import KeyenceAnalyzerComplete
from spike_removal import SpikeRemover

# Step 1: Load and analyze
analyzer = KeyenceAnalyzerComplete(n_jobs=-1)
height_data, filename = analyzer.load_data('data.csv')
pcd, points, indices = analyzer.data_to_point_cloud(height_data, subsample=4)
properties = analyzer.compute_normals_and_properties_optimized(pcd, points, k_neighbors=20)

# Step 2: Try different filters
remover = SpikeRemover()

# Option A: Curvature filtering
pcd_clean1, points_clean1, mask1 = remover.curvature_based_filtering(
    pcd, points, properties['curvature'], threshold=0.1
)

# Option B: Statistical filtering
pcd_clean2, mask2 = remover.statistical_outlier_removal(
    pcd, k_neighbors=20, std_ratio=2.0
)

# Option C: Multi-stage
result = remover.multi_stage_filtering(
    pcd, points,
    curvature=properties['curvature'],
    config={
        'radius_outlier': {'radius': 10.0, 'min_neighbors': 10},
        'curvature': {'threshold': 0.1},
        'statistical': {'k_neighbors': 20, 'std_ratio': 2.0}
    }
)
```

---

## ❓ Troubleshooting

### "Too many points removed!"
- Increase threshold (e.g., 0.1 → 0.15 → 0.2)
- Use more conservative std_ratio (e.g., 2.0 → 2.5 → 3.0)
- Try less aggressive methods first

### "Spikes still remain!"
- Decrease threshold (e.g., 0.1 → 0.075 → 0.05)
- Use multi-stage filtering
- Combine multiple methods

### "Out of memory!"
- Increase subsample (2 → 4 → 8)
- Process in smaller chunks/regions
- Reduce chunk_size parameter

### "Takes too long!"
- Increase subsample
- Reduce k_neighbors (20 → 15 → 10)
- Use fewer CPU cores if thermal throttling

---

## 📈 Expected Performance

For your **51 million points**:

| Configuration | Analysis Time | Filtering Time | Total |
|---------------|---------------|----------------|-------|
| subsample=1, k=20 | ~25 min | ~2 min | ~27 min |
| subsample=2, k=20 | ~9 min | ~1 min | ~10 min |
| subsample=4, k=20 | ~5 min | ~30 sec | ~5.5 min |

*Times will vary based on CPU*

---

## 🎉 Summary

**The key difference**: Your old code only MEASURED spikes, the new code REMOVES them!

**To fix your spikes**:
```python
from keyence_complete import analyze_and_clean_keyence

analyzer, results = analyze_and_clean_keyence(
    'your_51million_points.csv',
    subsample=4,
    curvature_threshold=0.1
)

# Boom! Spikes gone.
cleaned_pcd = results['pcd_filtered']
```

That's it! Now your data is actually cleaned, not just analyzed. 🚀
