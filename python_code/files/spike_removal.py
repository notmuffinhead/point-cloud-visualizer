import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import open3d as o3d
import time


class SpikeRemover:
    """
    Comprehensive spike removal toolkit for point cloud data.
    Provides multiple filtering methods to remove outliers, spikes, and noise.
    """
    
    def __init__(self):
        """Initialize spike remover."""
        print("✅ Spike Remover initialized")
    
    def statistical_outlier_removal(self, pcd, k_neighbors=20, std_ratio=2.0):
        """
        Remove statistical outliers based on neighbor distance.
        
        Args:
            pcd: Open3D PointCloud
            k_neighbors: Number of neighbors to consider (20-50 typical)
            std_ratio: Standard deviation multiplier (2.0-3.0 typical)
                      - 2.0: More aggressive (~5% removed)
                      - 3.0: More conservative (~1% removed)
        
        Returns:
            cleaned_pcd: Filtered point cloud
            inlier_mask: Boolean mask of kept points
        """
        print(f"\n🔍 Statistical Outlier Removal (k={k_neighbors}, std={std_ratio})")
        print(f"   Original points: {len(pcd.points):,}")
        
        t0 = time.time()
        cleaned_pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=k_neighbors,
            std_ratio=std_ratio
        )
        t1 = time.time()
        
        removed = len(pcd.points) - len(cleaned_pcd.points)
        percent_removed = 100 * removed / len(pcd.points)
        
        print(f"   Filtered points: {len(cleaned_pcd.points):,}")
        print(f"   ❌ Removed: {removed:,} ({percent_removed:.2f}%)")
        print(f"   ⏱️  Time: {t1-t0:.2f}s")
        
        # Create boolean mask
        inlier_mask = np.zeros(len(pcd.points), dtype=bool)
        inlier_mask[inlier_indices] = True
        
        return cleaned_pcd, inlier_mask
    
    def radius_outlier_removal(self, pcd, radius=10.0, min_neighbors=10):
        """
        Remove outliers based on neighbor density within radius.
        
        Args:
            pcd: Open3D PointCloud
            radius: Search radius in micrometers (2-5x pixel pitch typical)
            min_neighbors: Minimum neighbors required (5-20 typical)
        
        Returns:
            cleaned_pcd: Filtered point cloud
            inlier_mask: Boolean mask of kept points
        """
        print(f"\n🔍 Radius Outlier Removal (radius={radius}μm, min_neighbors={min_neighbors})")
        print(f"   Original points: {len(pcd.points):,}")
        
        t0 = time.time()
        cleaned_pcd, inlier_indices = pcd.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius
        )
        t1 = time.time()
        
        removed = len(pcd.points) - len(cleaned_pcd.points)
        percent_removed = 100 * removed / len(pcd.points)
        
        print(f"   Filtered points: {len(cleaned_pcd.points):,}")
        print(f"   ❌ Removed: {removed:,} ({percent_removed:.2f}%)")
        print(f"   ⏱️  Time: {t1-t0:.2f}s")
        
        # Create boolean mask
        inlier_mask = np.zeros(len(pcd.points), dtype=bool)
        inlier_mask[inlier_indices] = True
        
        return cleaned_pcd, inlier_mask
    
    def curvature_based_filtering(self, pcd, points, curvature, threshold=0.1):
        """
        Remove high-curvature points (spikes, sharp features).
        
        Args:
            pcd: Open3D PointCloud
            points: Nx3 numpy array
            curvature: N-length array of curvature values (from your analysis)
            threshold: Maximum allowed curvature (0.05-0.2 typical)
                      - 0.05: Very aggressive (removes subtle spikes)
                      - 0.1: Moderate (good default)
                      - 0.2: Conservative (only removes sharp spikes)
        
        Returns:
            cleaned_pcd: Filtered point cloud
            cleaned_points: Filtered points array
            inlier_mask: Boolean mask of kept points
        """
        print(f"\n🔍 Curvature-Based Filtering (threshold={threshold})")
        print(f"   Original points: {len(points):,}")
        print(f"   Curvature range: [{np.min(curvature):.6f}, {np.max(curvature):.6f}]")
        print(f"   Curvature mean: {np.mean(curvature):.6f}")
        
        t0 = time.time()
        
        # Keep points with curvature below threshold
        inlier_mask = curvature < threshold
        
        cleaned_points = points[inlier_mask]
        
        # Create new point cloud
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        
        t1 = time.time()
        
        removed = len(points) - len(cleaned_points)
        percent_removed = 100 * removed / len(points)
        
        print(f"   Filtered points: {len(cleaned_points):,}")
        print(f"   ❌ Removed: {removed:,} ({percent_removed:.2f}%)")
        print(f"   ⏱️  Time: {t1-t0:.2f}s")
        
        return cleaned_pcd, cleaned_points, inlier_mask
    
    def planarity_based_filtering(self, pcd, points, planarity, threshold=0.5):
        """
        Keep only planar regions (remove non-planar spikes/edges).
        
        Args:
            pcd: Open3D PointCloud
            points: Nx3 numpy array
            planarity: N-length array of planarity values (from your analysis)
            threshold: Minimum required planarity (0.5-0.8 typical)
                      - Higher threshold = keep only very flat regions
        
        Returns:
            cleaned_pcd: Filtered point cloud
            cleaned_points: Filtered points array
            inlier_mask: Boolean mask of kept points
        """
        print(f"\n🔍 Planarity-Based Filtering (threshold={threshold})")
        print(f"   Original points: {len(points):,}")
        print(f"   Planarity range: [{np.min(planarity):.4f}, {np.max(planarity):.4f}]")
        print(f"   Planarity mean: {np.mean(planarity):.4f}")
        
        t0 = time.time()
        
        # Keep points with planarity above threshold
        inlier_mask = planarity > threshold
        
        cleaned_points = points[inlier_mask]
        
        # Create new point cloud
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        
        t1 = time.time()
        
        removed = len(points) - len(cleaned_points)
        percent_removed = 100 * removed / len(points)
        
        print(f"   Filtered points: {len(cleaned_points):,}")
        print(f"   ❌ Removed: {removed:,} ({percent_removed:.2f}%)")
        print(f"   ⏱️  Time: {t1-t0:.2f}s")
        
        return cleaned_pcd, cleaned_points, inlier_mask
    
    def height_based_filtering(self, pcd, points, percentile_low=1, percentile_high=99):
        """
        Remove extreme height values (Z-filtering).
        
        Args:
            pcd: Open3D PointCloud
            points: Nx3 numpy array
            percentile_low: Lower percentile to keep (1-5 typical)
            percentile_high: Upper percentile to keep (95-99 typical)
        
        Returns:
            cleaned_pcd: Filtered point cloud
            cleaned_points: Filtered points array
            inlier_mask: Boolean mask of kept points
        """
        print(f"\n🔍 Height-Based Filtering (keep {percentile_low}%-{percentile_high}%)")
        print(f"   Original points: {len(points):,}")
        
        t0 = time.time()
        
        z_values = points[:, 2]
        z_min = np.percentile(z_values, percentile_low)
        z_max = np.percentile(z_values, percentile_high)
        
        print(f"   Height range: [{np.min(z_values):.2f}, {np.max(z_values):.2f}] μm")
        print(f"   Keeping: [{z_min:.2f}, {z_max:.2f}] μm")
        
        # Keep points within percentile range
        inlier_mask = (z_values >= z_min) & (z_values <= z_max)
        
        cleaned_points = points[inlier_mask]
        
        # Create new point cloud
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        
        t1 = time.time()
        
        removed = len(points) - len(cleaned_points)
        percent_removed = 100 * removed / len(points)
        
        print(f"   Filtered points: {len(cleaned_points):,}")
        print(f"   ❌ Removed: {removed:,} ({percent_removed:.2f}%)")
        print(f"   ⏱️  Time: {t1-t0:.2f}s")
        
        return cleaned_pcd, cleaned_points, inlier_mask
    
    def normal_based_filtering(self, pcd, points, normals, expected_direction=np.array([0, 0, 1]), 
                              angle_threshold=30):
        """
        Remove points with abnormal normal directions.
        
        Args:
            pcd: Open3D PointCloud
            points: Nx3 numpy array
            normals: Nx3 array of surface normals
            expected_direction: Expected normal direction [x, y, z] (default: [0,0,1] = upward)
            angle_threshold: Maximum angle deviation in degrees (20-45 typical)
        
        Returns:
            cleaned_pcd: Filtered point cloud
            cleaned_points: Filtered points array
            inlier_mask: Boolean mask of kept points
        """
        print(f"\n🔍 Normal-Based Filtering (angle threshold={angle_threshold}°)")
        print(f"   Original points: {len(points):,}")
        print(f"   Expected direction: {expected_direction}")
        
        t0 = time.time()
        
        # Normalize expected direction
        expected_direction = expected_direction / np.linalg.norm(expected_direction)
        
        # Compute dot product (cosine of angle)
        dot_products = np.dot(normals, expected_direction)
        
        # Convert to angles
        angles = np.arccos(np.clip(dot_products, -1, 1)) * 180 / np.pi
        
        print(f"   Angle range: [{np.min(angles):.1f}°, {np.max(angles):.1f}°]")
        print(f"   Angle mean: {np.mean(angles):.1f}°")
        
        # Keep points with small angle deviation
        inlier_mask = angles < angle_threshold
        
        cleaned_points = points[inlier_mask]
        
        # Create new point cloud
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        
        t1 = time.time()
        
        removed = len(points) - len(cleaned_points)
        percent_removed = 100 * removed / len(points)
        
        print(f"   Filtered points: {len(cleaned_points):,}")
        print(f"   ❌ Removed: {removed:,} ({percent_removed:.2f}%)")
        print(f"   ⏱️  Time: {t1-t0:.2f}s")
        
        return cleaned_pcd, cleaned_points, inlier_mask
    
    def multi_stage_filtering(self, pcd, points, curvature=None, planarity=None, 
                             normals=None, config=None):
        """
        Apply multiple filtering stages in sequence.
        
        Args:
            pcd: Open3D PointCloud
            points: Nx3 numpy array
            curvature: Optional curvature array
            planarity: Optional planarity array
            normals: Optional normals array
            config: Dictionary with filtering parameters
        
        Returns:
            dict with cleaned data and statistics
        """
        print("\n" + "="*70)
        print("🔧 MULTI-STAGE SPIKE REMOVAL PIPELINE")
        print("="*70)
        
        if config is None:
            config = {
                'radius_outlier': {'radius': 10.0, 'min_neighbors': 10},
                'curvature': {'threshold': 0.1},
                'statistical': {'k_neighbors': 20, 'std_ratio': 2.0}
            }
        
        original_count = len(points)
        current_pcd = pcd
        current_points = points
        cumulative_mask = np.ones(len(points), dtype=bool)
        
        stages = []
        
        # Stage 1: Radius outlier removal (if configured)
        if 'radius_outlier' in config:
            print("\n📍 STAGE 1: Radius Outlier Removal")
            current_pcd, mask = self.radius_outlier_removal(
                current_pcd, **config['radius_outlier']
            )
            current_points = np.asarray(current_pcd.points)
            cumulative_mask[cumulative_mask] = mask
            stages.append(('radius', len(current_points)))
        
        # Stage 2: Curvature filtering (if data provided)
        if curvature is not None and 'curvature' in config:
            print("\n📍 STAGE 2: Curvature-Based Filtering")
            # Filter curvature array to match current points
            current_curvature = curvature[cumulative_mask]
            current_pcd, current_points, mask = self.curvature_based_filtering(
                current_pcd, current_points, current_curvature, **config['curvature']
            )
            # Update cumulative mask
            temp_mask = np.zeros(len(cumulative_mask), dtype=bool)
            temp_mask[cumulative_mask] = mask
            cumulative_mask = temp_mask
            stages.append(('curvature', len(current_points)))
        
        # Stage 3: Statistical outlier removal (if configured)
        if 'statistical' in config:
            print("\n📍 STAGE 3: Statistical Outlier Removal")
            current_pcd, mask = self.statistical_outlier_removal(
                current_pcd, **config['statistical']
            )
            current_points = np.asarray(current_pcd.points)
            temp_mask = np.zeros(len(cumulative_mask), dtype=bool)
            temp_mask[cumulative_mask] = mask
            cumulative_mask = temp_mask
            stages.append(('statistical', len(current_points)))
        
        # Final summary
        final_count = len(current_points)
        total_removed = original_count - final_count
        percent_removed = 100 * total_removed / original_count
        
        print("\n" + "="*70)
        print("✅ FILTERING COMPLETE - SUMMARY")
        print("="*70)
        print(f"   Original points:  {original_count:,}")
        print(f"   Filtered points:  {final_count:,}")
        print(f"   ❌ Total removed: {total_removed:,} ({percent_removed:.2f}%)")
        print(f"\n   Stage breakdown:")
        prev_count = original_count
        for stage_name, stage_count in stages:
            removed = prev_count - stage_count
            print(f"     {stage_name}: {removed:,} points removed")
            prev_count = stage_count
        
        return {
            'pcd': current_pcd,
            'points': current_points,
            'mask': cumulative_mask,
            'original_count': original_count,
            'final_count': final_count,
            'removed_count': total_removed,
            'percent_removed': percent_removed,
            'stages': stages
        }
    
    def visualize_filtering(self, original_points, filtered_points, title="Filtering Result"):
        """
        Create before/after visualization of filtering.
        
        Args:
            original_points: Nx3 original points
            filtered_points: Mx3 filtered points
            title: Plot title
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Before Filtering', 'After Filtering'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # Before
        fig.add_trace(
            go.Scatter3d(
                x=original_points[:, 0],
                y=original_points[:, 1],
                z=original_points[:, 2],
                mode='markers',
                marker=dict(size=1, color=original_points[:, 2], colorscale='Viridis'),
                name='Original'
            ),
            row=1, col=1
        )
        
        # After
        fig.add_trace(
            go.Scatter3d(
                x=filtered_points[:, 0],
                y=filtered_points[:, 1],
                z=filtered_points[:, 2],
                mode='markers',
                marker=dict(size=1, color=filtered_points[:, 2], colorscale='Viridis'),
                name='Filtered'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            scene=dict(aspectmode='data'),
            scene2=dict(aspectmode='data'),
            width=1400,
            height=600
        )
        
        fig.show()
        print(f"✅ Visualization displayed")


def quick_spike_removal(pcd, points, method='curvature', curvature=None, **kwargs):
    """
    Quick spike removal with common presets.
    
    Args:
        pcd: Open3D PointCloud
        points: Nx3 array
        method: 'statistical', 'radius', 'curvature', 'height', or 'multi'
        curvature: Required if method='curvature'
        **kwargs: Method-specific parameters
    
    Returns:
        cleaned_pcd, cleaned_points, inlier_mask
    """
    remover = SpikeRemover()
    
    if method == 'statistical':
        k = kwargs.get('k_neighbors', 20)
        std = kwargs.get('std_ratio', 2.0)
        return remover.statistical_outlier_removal(pcd, k, std)
    
    elif method == 'radius':
        radius = kwargs.get('radius', 10.0)
        min_n = kwargs.get('min_neighbors', 10)
        return remover.radius_outlier_removal(pcd, radius, min_n)
    
    elif method == 'curvature':
        if curvature is None:
            raise ValueError("Curvature array required for curvature-based filtering")
        threshold = kwargs.get('threshold', 0.1)
        return remover.curvature_based_filtering(pcd, points, curvature, threshold)
    
    elif method == 'height':
        low = kwargs.get('percentile_low', 1)
        high = kwargs.get('percentile_high', 99)
        return remover.height_based_filtering(pcd, points, low, high)
    
    elif method == 'multi':
        return remover.multi_stage_filtering(pcd, points, curvature=curvature, **kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# Example usage
if __name__ == "__main__":
    print("🔧 Spike Removal Toolkit")
    print("="*60)
    print("\nAvailable methods:")
    print("  - statistical_outlier_removal")
    print("  - radius_outlier_removal")
    print("  - curvature_based_filtering")
    print("  - planarity_based_filtering")
    print("  - height_based_filtering")
    print("  - normal_based_filtering")
    print("  - multi_stage_filtering")
    print("\nSee SPIKE_REMOVAL_GUIDE.md for detailed usage instructions")
