"""
BATCH PROCESSING: Analyze and clean ALL CSV files in a folder
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
from numba import jit, prange
import time

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    raise ImportError("Open3D is required")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from spike_removal import SpikeRemover


@jit(nopython=True, parallel=True, fastmath=True)
def compute_covariance_batch_numba(points, neighbor_indices, neighbor_counts, max_neighbors):
    """Numba-accelerated covariance computation."""
    n_points = points.shape[0]
    planarity = np.zeros(n_points)
    curvature = np.zeros(n_points)
    eigenvalues_all = np.zeros((n_points, 3))
    
    for i in prange(n_points):
        k = neighbor_counts[i]
        
        if k < 3:
            continue
        
        neighbor_pts = np.zeros((k, 3))
        for j in range(k):
            idx = neighbor_indices[i, j]
            if idx >= 0:
                neighbor_pts[j] = points[idx]
        
        centroid = np.zeros(3)
        for j in range(k):
            centroid += neighbor_pts[j]
        centroid /= k
        
        for j in range(k):
            neighbor_pts[j] -= centroid
        
        cov = np.zeros((3, 3))
        for j in range(k):
            for d1 in range(3):
                for d2 in range(3):
                    cov[d1, d2] += neighbor_pts[j, d1] * neighbor_pts[j, d2]
        cov /= k
        
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues_all[i] = eigenvalues
        
        if eigenvalues[0] > 1e-10:
            planarity[i] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        
        total = eigenvalues[0] + eigenvalues[1] + eigenvalues[2]
        if total > 1e-10:
            curvature[i] = eigenvalues[2] / total
    
    return planarity, curvature, eigenvalues_all


class KeyenceAnalyzerComplete:
    """
    COMPLETE solution: Fast analysis + Spike removal + Batch processing
    """
    
    def __init__(self, n_jobs=-1):
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D is required")
        
        self.invalid_value = -99999.9999
        self.pixel_pitch_x = 2.5
        self.pixel_pitch_y = 2.5
        self.n_jobs = n_jobs
        self.spike_remover = SpikeRemover()
        self.results = {}
        
        print(f"✅ COMPLETE Keyence Analyzer initialized")
        print(f"   - Fast computation: {n_jobs} CPU cores")
        print(f"   - Spike removal: Multiple methods available")
    
    def load_data(self, filepath):
        """Load CSV data."""
        filename = os.path.basename(filepath)
        print(f"\n📁 Loading: {filename}")
        
        full_df = pd.read_csv(filepath, header=None)
        full_data = full_df.values.copy()
        
        invalid_mask = full_data == self.invalid_value
        full_data[invalid_mask] = np.nan
        
        num_invalid = np.sum(invalid_mask)
        total_points = full_data.size
        
        print(f"📋 Shape: {full_data.shape[0]} × {full_data.shape[1]}")
        print(f"📊 Valid: {total_points - num_invalid:,}/{total_points:,} ({100*(1-num_invalid/total_points):.1f}%)")
        
        return full_data, filename
    
    def data_to_point_cloud(self, height_data, subsample=1):
        """Convert height map to point cloud (vectorized)."""
        rows, cols = height_data.shape
        
        if subsample > 1:
            height_data = height_data[::subsample, ::subsample]
            rows, cols = height_data.shape
        
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        
        z_mm = height_data.ravel()
        valid_mask = ~np.isnan(z_mm)
        
        z_mm = z_mm[valid_mask]
        x_pixels = x_coords.ravel()[valid_mask]
        y_pixels = y_coords.ravel()[valid_mask]
        
        x_um = x_pixels * self.pixel_pitch_x * subsample
        y_um = y_pixels * self.pixel_pitch_y * subsample
        z_um = z_mm * 1000
        
        points = np.column_stack([x_um, y_um, z_um])
        valid_indices = np.column_stack([y_pixels * subsample, x_pixels * subsample])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"📍 Point cloud: {len(points):,} valid points")
        return pcd, points, valid_indices
    
    def compute_normals_and_properties_optimized(self, pcd, points, k_neighbors=20, 
                                                  chunk_size=100000):
        """Optimized computation with Numba (fixed for Windows pickling)."""
        print(f"\n🚀 OPTIMIZED computation")
        print(f"   Method: Sequential KNN + Numba JIT parallel")
        print(f"   Jobs: {self.n_jobs if self.n_jobs > 0 else 'all cores'}")
        
        t0 = time.time()
        
        print(f"\n🔬 Step 1/3: Computing normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0, 0, 1e6])
        )
        normals = np.asarray(pcd.normals)
        t1 = time.time()
        print(f"   ✅ Done in {t1-t0:.2f}s")
        
        print(f"\n🔬 Step 2/3: Finding neighbors (sequential - Windows compatible)...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        n_points = len(points)
        
        # Pre-allocate arrays
        neighbor_indices = np.full((n_points, k_neighbors), -1, dtype=np.int32)
        neighbor_counts = np.zeros(n_points, dtype=np.int32)
        
        # Sequential neighbor queries (still fast with Open3D's optimized KDTree)
        print_interval = max(1, n_points // 20)  # Print progress ~20 times
        for i in range(n_points):
            if i % print_interval == 0 and i > 0:
                print(f"   Progress: {i:,}/{n_points:,} ({100*i/n_points:.1f}%)")
            
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_neighbors)
            neighbor_counts[i] = k
            neighbor_indices[i, :len(idx)] = idx
        
        t2 = time.time()
        print(f"   ✅ Done in {t2-t1:.2f}s")
        
        print(f"\n🔬 Step 3/3: Computing planarity & curvature (Numba parallel)...")
        planarity, curvature, eigenvalues_all = compute_covariance_batch_numba(
            points, neighbor_indices, neighbor_counts, k_neighbors
        )
        
        t3 = time.time()
        print(f"   ✅ Done in {t3-t2:.2f}s")
        
        total_time = t3 - t0
        print(f"\n✅ TOTAL TIME: {total_time:.2f}s ({n_points/total_time:,.0f} pts/sec)")
        
        return {
            'normals': normals,
            'planarity': planarity,
            'curvature': curvature,
            'eigenvalues': eigenvalues_all
        }
    
    def analyze_and_filter(self, filepath, k_neighbors=20, subsample=2,
                          filter_spikes=True, filter_method='curvature',
                          filter_params=None, visualize=False, save_output=True,
                          output_dir='cleaned_output'):
        """
        COMPLETE WORKFLOW: Analyze + Remove spikes + Save results
        
        Args:
            filepath: Path to CSV file
            k_neighbors: Neighbors for PCA
            subsample: Point reduction factor
            filter_spikes: Whether to filter spikes
            filter_method: 'curvature', 'statistical', 'radius', 'multi'
            filter_params: Dict of filtering parameters
            visualize: Create visualizations (set False for batch to save time)
            save_output: Save cleaned point cloud to file
            output_dir: Directory to save output files
            
        Returns:
            dict with original and filtered results
        """
        print("=" * 70)
        print("🚀 COMPLETE ANALYSIS + SPIKE REMOVAL")
        print("=" * 70)
        
        # Load data
        height_data, filename = self.load_data(filepath)
        
        # Convert to point cloud
        pcd_original, points_original, valid_indices = self.data_to_point_cloud(
            height_data, subsample
        )
        
        # Compute properties
        properties = self.compute_normals_and_properties_optimized(
            pcd_original, points_original, k_neighbors
        )
        
        normals = properties['normals']
        planarity = properties['planarity']
        curvature = properties['curvature']
        
        print(f"\n📊 ORIGINAL DATA STATISTICS:")
        print(f"   Points: {len(points_original):,}")
        print(f"   Curvature: mean={np.mean(curvature):.6f}, max={np.max(curvature):.6f}")
        print(f"   Planarity: mean={np.mean(planarity):.4f}, min={np.min(planarity):.4f}")
        
        # Spike removal
        if filter_spikes:
            print("\n" + "="*70)
            print("🔧 SPIKE REMOVAL")
            print("="*70)
            
            if filter_params is None:
                if filter_method == 'curvature':
                    filter_params = {'threshold': 0.1}
                elif filter_method == 'statistical':
                    filter_params = {'k_neighbors': 20, 'std_ratio': 2.0}
                elif filter_method == 'radius':
                    filter_params = {'radius': 10.0, 'min_neighbors': 10}
                elif filter_method == 'multi':
                    filter_params = {
                        'config': {
                            'radius_outlier': {'radius': 10.0, 'min_neighbors': 10},
                            'curvature': {'threshold': 0.1},
                            'statistical': {'k_neighbors': 20, 'std_ratio': 2.0}
                        },
                        'curvature': curvature
                    }
            
            # Apply filtering
            if filter_method == 'curvature':
                pcd_filtered, points_filtered, mask = self.spike_remover.curvature_based_filtering(
                    pcd_original, points_original, curvature, **filter_params
                )
                normals_filtered = normals[mask]
                planarity_filtered = planarity[mask]
                curvature_filtered = curvature[mask]
                valid_indices_filtered = valid_indices[mask]
                
            elif filter_method == 'statistical':
                pcd_filtered, mask = self.spike_remover.statistical_outlier_removal(
                    pcd_original, **filter_params
                )
                points_filtered = np.asarray(pcd_filtered.points)
                normals_filtered = normals[mask]
                planarity_filtered = planarity[mask]
                curvature_filtered = curvature[mask]
                valid_indices_filtered = valid_indices[mask]
                
            elif filter_method == 'radius':
                pcd_filtered, mask = self.spike_remover.radius_outlier_removal(
                    pcd_original, **filter_params
                )
                points_filtered = np.asarray(pcd_filtered.points)
                normals_filtered = normals[mask]
                planarity_filtered = planarity[mask]
                curvature_filtered = curvature[mask]
                valid_indices_filtered = valid_indices[mask]
                
            elif filter_method == 'multi':
                result = self.spike_remover.multi_stage_filtering(
                    pcd_original, points_original, **filter_params
                )
                pcd_filtered = result['pcd']
                points_filtered = result['points']
                mask = result['mask']
                normals_filtered = normals[mask]
                planarity_filtered = planarity[mask]
                curvature_filtered = curvature[mask]
                valid_indices_filtered = valid_indices[mask]
            
            else:
                raise ValueError(f"Unknown filter method: {filter_method}")
            
            print(f"\n📊 FILTERED DATA STATISTICS:")
            print(f"   Points: {len(points_filtered):,}")
            print(f"   Curvature: mean={np.mean(curvature_filtered):.6f}, max={np.max(curvature_filtered):.6f}")
            print(f"   Planarity: mean={np.mean(planarity_filtered):.4f}, min={np.min(planarity_filtered):.4f}")
            
            # Save output
            if save_output:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_cleaned.ply")
                o3d.io.write_point_cloud(output_path, pcd_filtered)
                print(f"\n💾 Saved cleaned point cloud: {output_path}")
            
            # Visualization (optional)
            if visualize:
                self.visualize_before_after(
                    points_original, points_filtered, 
                    curvature, curvature_filtered,
                    f"Spike Removal: {filter_method}"
                )
        
        else:
            pcd_filtered = pcd_original
            points_filtered = points_original
            normals_filtered = normals
            planarity_filtered = planarity
            curvature_filtered = curvature
            valid_indices_filtered = valid_indices
            mask = np.ones(len(points_original), dtype=bool)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 70)
        
        return {
            'filename': filename,
            'pcd_original': pcd_original,
            'points_original': points_original,
            'normals_original': normals,
            'planarity_original': planarity,
            'curvature_original': curvature,
            'valid_indices_original': valid_indices,
            'pcd_filtered': pcd_filtered,
            'points_filtered': points_filtered,
            'normals_filtered': normals_filtered,
            'planarity_filtered': planarity_filtered,
            'curvature_filtered': curvature_filtered,
            'valid_indices_filtered': valid_indices_filtered,
            'filter_mask': mask,
            'filter_method': filter_method if filter_spikes else None,
            'full_shape': height_data.shape
        }
    
    def visualize_before_after(self, points_before, points_after, 
                              curvature_before, curvature_after, title):
        """Create clean 3D surface plots (before and after)."""
        
        print("\n📊 Creating 3D visualizations...")
        
        # Sample points for display (for performance)
        max_display_points = 250000
        
        if len(points_before) > max_display_points:
            subsample = max(1, len(points_before) // max_display_points)
            idx_before = np.arange(0, len(points_before), subsample)
            idx_after = np.arange(0, len(points_after), subsample)
        else:
            idx_before = np.arange(len(points_before))
            idx_after = np.arange(len(points_after))
        
        # Plot 1: ORIGINAL (with spikes)
        fig_before = go.Figure(data=[
            go.Scatter3d(
                x=points_before[idx_before, 0],
                y=points_before[idx_before, 1],
                z=points_before[idx_before, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=points_before[idx_before, 2],  # Color by height
                    colorscale='Viridis',
                    colorbar=dict(title="Height (μm)"),
                    showscale=True
                ),
                name='Original Surface'
            )
        ])
        
        fig_before.update_layout(
            title=f"ORIGINAL Surface (With Spikes) - {len(points_before):,} points",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Height (μm)",
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        # Plot 2: CLEANED (spikes removed)
        fig_after = go.Figure(data=[
            go.Scatter3d(
                x=points_after[idx_after, 0],
                y=points_after[idx_after, 1],
                z=points_after[idx_after, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=points_after[idx_after, 2],  # Color by height
                    colorscale='Viridis',
                    colorbar=dict(title="Height (μm)"),
                    showscale=True
                ),
                name='Cleaned Surface'
            )
        ])
        
        fig_after.update_layout(
            title=f"CLEANED Surface (Spikes Removed) - {len(points_after):,} points",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Height (μm)",
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        # Show both plots (opens in separate browser tabs)
        fig_before.show()
        fig_after.show()
        
        removed = len(points_before) - len(points_after)
        print(f"✅ 3D visualizations displayed")
        print(f"   Original: {len(points_before):,} points")
        print(f"   Cleaned:  {len(points_after):,} points")
        print(f"   Removed:  {removed:,} spikes ({100*removed/len(points_before):.2f}%)")
    
    def analyze_all_files(self, folder_path="KEYENCE_DATASET", k_neighbors=20,
                         subsample=2, filter_spikes=True, filter_method='curvature',
                         filter_params=None, visualize=False, save_outputs=True,
                         output_dir='cleaned_output'):
        """
        BATCH PROCESSING: Analyze and clean all CSV files in a folder.
        
        Args:
            folder_path: Path to folder with CSV files
            k_neighbors: Neighbors for PCA
            subsample: Point reduction factor
            filter_spikes: Enable spike removal
            filter_method: Filtering method to use
            filter_params: Parameters for filtering
            visualize: Show plots (False recommended for batch)
            save_outputs: Save cleaned files
            output_dir: Where to save output files
            
        Returns:
            dict: Results for all files
        """
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {folder_path}")
            return {}
        
        print(f"🚀 BATCH ANALYSIS + SPIKE REMOVAL")
        print(f"📂 Folder: {folder_path}")
        print(f"📄 Found {len(csv_files)} CSV files")
        print(f"🔧 Filter method: {filter_method if filter_spikes else 'None'}")
        print(f"💾 Output dir: {output_dir}")
        print("=" * 70)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*70}")
            print(f"PROCESSING FILE {i} OF {len(csv_files)}")
            print(f"{'='*70}")
            
            try:
                result = self.analyze_and_filter(
                    csv_file,
                    k_neighbors=k_neighbors,
                    subsample=subsample,
                    filter_spikes=filter_spikes,
                    filter_method=filter_method,
                    filter_params=filter_params,
                    visualize=visualize,
                    save_output=save_outputs,
                    output_dir=output_dir
                )
                
                # Store summary
                self.results[result['filename']] = {
                    'status': 'success',
                    'original_points': len(result['points_original']),
                    'filtered_points': len(result['points_filtered']),
                    'removed_points': len(result['points_original']) - len(result['points_filtered']),
                    'percent_removed': 100 * (1 - len(result['points_filtered']) / len(result['points_original'])),
                    'mean_curvature_before': np.mean(result['curvature_original']),
                    'mean_curvature_after': np.mean(result['curvature_filtered']),
                }
                
            except Exception as e:
                filename = os.path.basename(csv_file)
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                self.results[filename] = {'status': 'failed', 'error': str(e)}
            
            if i < len(csv_files):
                input(f"\n⏸️  Press Enter to continue to next file ({i+1}/{len(csv_files)})...")
        
        # Print summary
        print(f"\n{'='*70}")
        print("🎉 BATCH PROCESSING COMPLETE!")
        print(f"{'='*70}")
        
        successful = [r for r in self.results.values() if r.get('status') == 'success']
        failed = [r for r in self.results.values() if r.get('status') == 'failed']
        
        print(f"\n✅ Successfully processed: {len(successful)} files")
        print(f"❌ Failed: {len(failed)} files")
        
        if successful:
            print(f"\n📊 SUMMARY STATISTICS:")
            total_original = sum(r['original_points'] for r in successful)
            total_filtered = sum(r['filtered_points'] for r in successful)
            total_removed = sum(r['removed_points'] for r in successful)
            
            print(f"   Total points (original): {total_original:,}")
            print(f"   Total points (filtered): {total_filtered:,}")
            print(f"   Total removed: {total_removed:,} ({100*total_removed/total_original:.2f}%)")
            
            print(f"\n   Per-file breakdown:")
            for filename, stats in self.results.items():
                if stats['status'] == 'success':
                    print(f"   📄 {filename}:")
                    print(f"      {stats['original_points']:,} → {stats['filtered_points']:,} "
                          f"(-{stats['removed_points']:,}, {stats['percent_removed']:.1f}%)")
        
        return self.results


def quick_keyence_analysis_and_clean(folder_path="KEYENCE_DATASET", subsample=2,
                                    filter_method='curvature', curvature_threshold=0.1,
                                    k_neighbors=20, n_jobs=-1, visualize=False,
                                    output_dir='cleaned_output'):
    """
    ONE-LINE BATCH PROCESSING for entire folder!
    
    Args:
        folder_path: Folder with CSV files
        subsample: Point reduction (2=half, 4=quarter)
        filter_method: 'curvature', 'statistical', 'radius', 'multi'
        curvature_threshold: For curvature filtering
        k_neighbors: Neighbors for PCA
        n_jobs: CPU cores (-1 = all)
        visualize: Show plots (False recommended for batch)
        output_dir: Where to save cleaned files
    
    Returns:
        analyzer, results
    
    Example:
        # Process all files in folder
        analyzer, results = quick_keyence_analysis_and_clean('KEYENCE_DATASET')
        
        # More aggressive filtering
        analyzer, results = quick_keyence_analysis_and_clean(
            'KEYENCE_DATASET', 
            curvature_threshold=0.05
        )
    """
    analyzer = KeyenceAnalyzerComplete(n_jobs=n_jobs)
    
    filter_params = None
    if filter_method == 'curvature':
        filter_params = {'threshold': curvature_threshold}
    
    results = analyzer.analyze_all_files(
        folder_path=folder_path,
        k_neighbors=k_neighbors,
        subsample=subsample,
        filter_spikes=True,
        filter_method=filter_method,
        filter_params=filter_params,
        visualize=visualize,
        save_outputs=True,
        output_dir=output_dir
    )
    
    return analyzer, results


if __name__ == "__main__":
    print("🚀 BATCH PROCESSING: Analyze + Clean Entire Folders")
    print("="*60)
    print("\nQuick start:")
    print("  analyzer, results = quick_keyence_analysis_and_clean('KEYENCE_DATASET')")
    print("\nAll cleaned files will be saved to 'cleaned_output/' folder")