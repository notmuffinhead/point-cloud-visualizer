# Spike Removal Methods for Point Cloud Data

## 🔍 Why PCA Didn't Remove Spikes

**Your current code only ANALYZES the surface** - it computes:
- Surface normals
- Planarity (how flat each region is)
- Curvature (how much the surface bends)

**But it NEVER modifies or filters the data!** It's like measuring a patient's temperature but not giving them medicine.

For **actual spike removal**, you need FILTERING methods.

---

## 🎯 Spike Removal Methods

### Method 1: Statistical Outlier Removal ⭐ BEST FOR RANDOM SPIKES
**Concept**: Remove points whose distance to neighbors is statistically abnormal

**How it works**:
- Compute average distance from each point to its k neighbors
- Calculate mean (μ) and standard deviation (σ) of all distances
- Remove points where distance > μ + n*σ (typically n=2 or 3)

**Best for**: 
- Random noise spikes
- Individual outlier points
- Measurement artifacts

**Pros**: Simple, effective for isolated spikes
**Cons**: May not work well for clustered spikes or systematic errors

---

### Method 2: Radius Outlier Removal ⭐ BEST FOR ISOLATED SPIKES
**Concept**: Remove points that have too few neighbors within a radius

**How it works**:
- For each point, count neighbors within radius R
- Remove points with < N neighbors

**Best for**:
- Isolated spikes that stick out from the surface
- Low-density outlier regions
- "Flying pixels" (common in laser scanners)

**Pros**: Very effective for sparse outliers
**Cons**: Requires tuning radius parameter

---

### Method 3: Curvature-Based Filtering ⭐ BEST FOR YOUR CASE (REAL SPIKES)
**Concept**: Remove high-curvature regions (sharp features)

**How it works**:
- Compute curvature for each point (you already have this!)
- Remove points where curvature > threshold
- Spikes have very high curvature

**Best for**:
- Actual physical spikes/protrusions
- Sharp edges you want to remove
- Burrs or manufacturing defects

**Pros**: Targets actual geometric spikes
**Cons**: May remove wanted sharp features

---

### Method 4: Moving Least Squares (MLS) Surface Reconstruction ⭐ SMOOTHING
**Concept**: Fit a smooth surface through the noisy data

**How it works**:
- For each point, fit a polynomial surface to local neighborhood
- Project point onto smooth surface
- Creates a smooth, denoised version

**Best for**:
- Preserving overall shape while removing noise
- When you want smoothing, not removal
- Continuous surface reconstruction

**Pros**: Preserves topology, looks nice
**Cons**: Computationally expensive, may lose fine details

---

### Method 5: RANSAC Plane Fitting ⭐ FOR PLANAR SURFACES
**Concept**: Fit planes robustly, ignore outliers

**How it works**:
- Randomly sample points to fit planes
- Count inliers (points near the plane)
- Keep the plane with most inliers
- Outliers are spikes

**Best for**:
- Flat surfaces with spikes
- Machined parts, metal sheets
- Dominant planar regions

**Pros**: Very robust to outliers
**Cons**: Assumes surface is mostly planar

---

### Method 6: Normal-Based Filtering
**Concept**: Remove points with abnormal normal directions

**How it works**:
- Compute normals (you already have this!)
- For predominantly horizontal surface, remove points where normal is far from vertical
- Or remove points whose normal differs significantly from neighbors

**Best for**:
- Spikes pointing in wrong direction
- Surface orientation consistency

**Pros**: Geometrically intuitive
**Cons**: Requires knowing expected normal direction

---

### Method 7: Height-Based Filtering (Z-filtering)
**Concept**: Remove points outside expected height range

**How it works**:
- Compute median or mean Z value
- Remove points where |Z - median| > threshold
- Or use percentile filtering (keep 95% of data)

**Best for**:
- Spikes that stick up significantly
- Known surface height range
- Simple, fast filtering

**Pros**: Very simple and fast
**Cons**: May remove valid high/low regions

---

### Method 8: Bilateral Filtering
**Concept**: Edge-preserving smoothing (like Photoshop's surface blur)

**How it works**:
- Smooth based on both spatial and intensity (height) similarity
- Preserves edges while smoothing flat regions

**Best for**:
- Preserving sharp features while removing noise
- Balancing smoothing and detail

**Pros**: Preserves edges
**Cons**: Complex parameter tuning

---

### Method 9: Morphological Operations
**Concept**: Opening (erosion + dilation) to remove spikes

**How it works**:
- Convert to voxel grid
- Apply morphological opening
- Removes small protrusions

**Best for**:
- Small spikes relative to surface features
- Grid-based operations

**Pros**: Effective for certain spike types
**Cons**: Requires voxelization (can be slow)

---

## 🎯 Recommended Approach for Your Data

Since you have **real physical spikes** (not just noise), I recommend:

### **Option A: Curvature + Statistical Combo** (Best for most cases)
1. Compute curvature (you have this!)
2. Remove high-curvature points (curvature > 0.1 or 0.2)
3. Then apply statistical outlier removal for remaining noise

### **Option B: Multi-Stage Pipeline** (Most robust)
1. **Radius outlier removal** - Remove isolated flying pixels
2. **Curvature filtering** - Remove spikes (high curvature)
3. **Statistical outlier removal** - Clean up remaining noise
4. **Optional: MLS smoothing** - Final smoothing pass

### **Option C: Height + Curvature** (Fast and simple)
1. Remove extreme Z values (above/below percentiles)
2. Remove high-curvature points
3. Done!

---

## ⚙️ Parameter Selection Guide

### Statistical Outlier Removal:
- `k_neighbors`: 20-50 (more = smoother threshold)
- `std_ratio`: 2.0-3.0 (lower = more aggressive)
  - 2.0: Remove ~5% of points
  - 3.0: Remove ~1% of points

### Radius Outlier Removal:
- `radius`: 2-5x your pixel pitch (for you: 5-12.5 μm)
- `min_neighbors`: 5-20 points

### Curvature Filtering:
- Threshold: 0.05-0.2 (examine your curvature histogram!)
- Start conservative (0.2), then lower if needed

### MLS Parameters:
- `search_radius`: 3-10x pixel pitch
- `polynomial_order`: 2 (quadratic)

---

## 🚨 Important Considerations

### Before Filtering:
1. **Visualize your data first!** Look at curvature/planarity histograms
2. **Understand your spikes**: Are they errors or real features?
3. **Test on a small region** before processing 51M points

### After Filtering:
1. **Check how many points were removed** (should be < 10% usually)
2. **Visualize before/after** to ensure you didn't over-filter
3. **Verify key features are preserved**

### Trade-offs:
- **Aggressive filtering** → Smooth surface, may lose detail
- **Conservative filtering** → Keep detail, may keep some spikes
- **Multiple passes** → Better results, slower processing

---

## 💡 Quick Decision Tree

```
Are spikes random individual points?
├─ YES → Use Statistical or Radius Outlier Removal
└─ NO (spikes are real protrusions)
    ├─ Want to REMOVE spikes? 
    │   └─ Use Curvature-Based Filtering
    └─ Want to SMOOTH spikes?
        └─ Use MLS Surface Reconstruction
```

---

Next: See the implementation code for all these methods!
