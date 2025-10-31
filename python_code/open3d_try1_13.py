import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("=" * 70)
    print("❌ ERROR: Open3D is not installed!")
    print("=" * 70)
    print("Open3D requires Python 3.7-3.11 (you have Python 3.13)")
    print("\nPlease see the installation guide for solutions.")
    print("=" * 70)
    raise ImportError("Open3D is required but not available. See guide for Python version solutions.")


class KeyenceAnalyzerOpen3D:
    """
    Keyence analyzer using Open3D for point cloud processing.
    Visualizations use Plotly for interactive web-based plots.
    
    Requirements:
    - Python 3.7-3.11
    - open3d
    - numpy, pandas, plotly
    
    Features:
    - Fast Open3D-based PCA plane fitting
    - Surface normal estimation
    - Curvature and planarity analysis
    - Interactive Plotly visualizations
    - Batch processing of multiple CSV files
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D is required for this analyzer")
        
        self.invalid_value = -99999.9999
        self.pixel_pitch_x = 2.5  # μm
        self.pixel_pitch_y = 2.5  # μm
        self.results = {}
        print("✅ Open3D analyzer initialized")
    
    def load_data(self, filepath):
        """Load and preprocess Keyence CSV data."""
        filename = os.path.basename(filepath)
        print(f"\n📁 Loading: {filename}")
        
        # Load CSV
        full_df = pd.read_csv(filepath, header=None)
        full_data = full_df.values.copy()
        
        # Replace invalid values with NaN
        invalid_mask = full_data == self.invalid_value
        full_data[invalid_mask] = np.nan
        
        num_invalid = np.sum(invalid_mask)
        total_points = full_data.size
        
        print(f"📋 Shape: {full_data.shape[0]} × {full_data.shape[1]}")
        print(f"📊 Valid points: {total_points - num_invalid:,}/{total_points:,} ({100*(1-num_invalid/total_points):.1f}%)")
        
        return full_data, filename
    
    def data_to_point_cloud(self, height_data, subsample=1):
        """
        Convert height map to XYZ point cloud.
        
        Args:
            height_data: 2D array of heights in mm
            subsample: Take every Nth point (1=all, 2=half, etc.)
            
        Returns:
            pcd: Open3D PointCloud object
            points: Nx3 numpy array of (x, y, z) in micrometers
            valid_indices: Nx2 array of original (row, col) indices
        """
        rows, cols = height_data.shape
        
        # Create subsampled grids
        row_idx = np.arange(0, rows, subsample)
        col_idx = np.arange(0, cols, subsample)
        
        # Build point lists
        points_list = []
        indices_list = []
        
        for i in row_idx:
            for j in col_idx:
                z_mm = height_data[i, j]
                if not np.isnan(z_mm):
                    x_um = j * self.pixel_pitch_x
                    y_um = i * self.pixel_pitch_y
                    z_um = z_mm * 1000  # mm to μm
                    points_list.append([x_um, y_um, z_um])
                    indices_list.append([i, j])
        
        points = np.array(points_list)
        valid_indices = np.array(indices_list)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"📍 Point cloud: {len(points):,} valid points")
        return pcd, points, valid_indices
    
    def compute_normals_and_properties(self, pcd, points, k_neighbors=20, radius=None):
        """
        Compute surface normals and geometric properties using Open3D.
        
        Args:
            pcd: Open3D PointCloud object
            points: Nx3 numpy array
            k_neighbors: Number of neighbors for normal estimation
            radius: Optional search radius instead of k-neighbors
            
        Returns:
            dict with normals, planarity, curvature, eigenvalues
        """
        print(f"\n🔬 Computing surface properties with Open3D (k={k_neighbors})...")
        
        # Estimate normals using Open3D
        if radius is not None:
            print(f"   Using hybrid search (radius={radius} μm, max_nn={k_neighbors})")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=k_neighbors
                )
            )
        else:
            print(f"   Using KNN search (k={k_neighbors})")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
            )
        
        # Orient normals consistently (toward +Z / "up")
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0, 0, 1e6])
        )
        
        normals = np.asarray(pcd.normals)
        print(f"✅ Normals computed: {len(normals)} vectors")
        
        # Compute covariance-based properties for each point
        print("🔬 Computing planarity and curvature...")
        
        # Build KDTree for neighbor queries
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        planarity = np.zeros(len(points))
        curvature = np.zeros(len(points))
        eigenvalues_all = np.zeros((len(points), 3))
        
        for i in range(len(points)):
            if i % 10000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{len(points):,} ({100*i/len(points):.1f}%)")
            
            # Find k nearest neighbors
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_neighbors)
            
            if k < 3:  # Need at least 3 points
                continue
            
            # Get neighbor coordinates
            neighbors = points[idx, :]
            
            # Center the neighborhood
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            
            # Compute covariance matrix
            cov = np.cov(centered.T)
            
            # Eigenvalue decomposition
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            eigenvalues_all[i] = eigenvalues
            
            # Planarity: (λ1 - λ2) / λ0
            # High value = planar surface
            if eigenvalues[0] > 1e-10:
                planarity[i] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
            
            # Curvature: λ2 / (λ0 + λ1 + λ2)
            # Measures how much surface bends
            total = np.sum(eigenvalues)
            if total > 1e-10:
                curvature[i] = eigenvalues[2] / total
        
        print("✅ Properties computed successfully!")
        
        return {
            'normals': normals,
            'planarity': planarity,
            'curvature': curvature,
            'eigenvalues': eigenvalues_all
        }
    
    def compute_surface_statistics(self, points, normals, planarity, curvature):
        """Compute summary statistics of surface properties."""
        print(f"\n📊 SURFACE ANALYSIS RESULTS")
        print("=" * 60)
        
        # Height statistics
        z_values = points[:, 2]
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        z_min = np.min(z_values)
        z_max = np.max(z_values)
        rms_roughness = np.sqrt(np.mean((z_values - z_mean)**2))
        
        print(f"📏 Height Statistics:")
        print(f"   Range: {z_min:.2f} to {z_max:.2f} μm (span: {z_max-z_min:.2f} μm)")
        print(f"   Mean: {z_mean:.2f} μm")
        print(f"   Std Dev: {z_std:.3f} μm")
        print(f"   RMS Roughness: {rms_roughness:.3f} μm")
        
        # Normal analysis
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        tilt_angle = np.arccos(np.clip(mean_normal[2], -1, 1)) * 180 / np.pi
        
        print(f"\n📐 Surface Orientation:")
        print(f"   Mean normal: [{mean_normal[0]:.4f}, {mean_normal[1]:.4f}, {mean_normal[2]:.4f}]")
        print(f"   Tilt from vertical: {tilt_angle:.2f}°")
        
        # Planarity statistics
        print(f"\n📏 Planarity (0=edge/ridge, 1=flat plane):")
        print(f"   Mean: {np.mean(planarity):.4f}")
        print(f"   Std: {np.std(planarity):.4f}")
        print(f"   Median: {np.median(planarity):.4f}")
        
        # Curvature statistics
        print(f"\n🌊 Curvature:")
        print(f"   Mean: {np.mean(curvature):.6f}")
        print(f"   Std: {np.std(curvature):.6f}")
        print(f"   Max: {np.max(curvature):.6f}")
        print(f"   95th percentile: {np.percentile(curvature, 95):.6f}")
        
        return {
            'z_mean': z_mean,
            'z_std': z_std,
            'z_range': z_max - z_min,
            'rms_roughness': rms_roughness,
            'mean_normal': mean_normal,
            'tilt_angle': tilt_angle,
            'mean_planarity': np.mean(planarity),
            'mean_curvature': np.mean(curvature),
            'max_curvature': np.max(curvature)
        }
    
    def plot_3d_surface_with_properties(self, points, valid_indices, property_values,
                                       property_name, full_shape, colorscale='Viridis'):
        """
        Create interactive 3D surface plot colored by property using Plotly.
        
        Args:
            points: Nx3 point cloud
            valid_indices: Nx2 array of (row, col) indices
            property_values: N array of property values for coloring
            property_name: Name for colorbar
            full_shape: Original (rows, cols) shape
            colorscale: Plotly colorscale name
        """
        print(f"📊 Creating 3D surface plot: {property_name}...")
        
        # Create property grid
        property_grid = np.full(full_shape, np.nan)
        for idx, val in zip(valid_indices, property_values):
            property_grid[idx[0], idx[1]] = val
        
        # Create coordinate grids
        rows, cols = full_shape
        x_coords = np.arange(cols) * self.pixel_pitch_x
        y_coords = np.arange(rows) * self.pixel_pitch_y
        
        # Reconstruct Z from points
        z_grid = np.full(full_shape, np.nan)
        for idx, pt in zip(valid_indices, points):
            z_grid[idx[0], idx[1]] = pt[2]
        
        # Create surface plot
        fig = go.Figure(data=[
            go.Surface(
                x=x_coords,
                y=y_coords,
                z=z_grid,
                surfacecolor=property_grid,
                colorscale=colorscale,
                colorbar=dict(title=property_name),
                name=property_name
            )
        ])
        
        fig.update_layout(
            title=f"3D Surface Colored by {property_name}",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Height (μm)",
                aspectmode='data'
            ),
            width=1100,
            height=800
        )
        
        fig.show()
        print(f"✅ Plot created!")
        return fig
    
    def plot_2d_heatmap(self, valid_indices, property_values, property_name, 
                       full_shape, colorscale='Viridis'):
        """Create 2D top-down heatmap of surface property."""
        print(f"📊 Creating 2D heatmap: {property_name}...")
        
        # Create property grid
        property_grid = np.full(full_shape, np.nan)
        for idx, val in zip(valid_indices, property_values):
            property_grid[idx[0], idx[1]] = val
        
        fig = go.Figure(data=go.Heatmap(
            z=property_grid,
            colorscale=colorscale,
            colorbar=dict(title=property_name)
        ))
        
        fig.update_layout(
            title=f"{property_name} - Top View",
            xaxis_title="Column Index",
            yaxis_title="Row Index",
            width=900,
            height=700
        )
        
        fig.show()
        print(f"✅ Heatmap created!")
        return fig
    
    def plot_property_histogram(self, property_values, property_name, bins=50):
        """Create histogram of property distribution."""
        print(f"📊 Creating histogram: {property_name}...")
        
        fig = go.Figure(data=[
            go.Histogram(
                x=property_values,
                nbinsx=bins,
                name=property_name
            )
        ])
        
        fig.update_layout(
            title=f"{property_name} Distribution",
            xaxis_title=property_name,
            yaxis_title="Count",
            width=800,
            height=500,
            showlegend=False
        )
        
        fig.show()
        print(f"✅ Histogram created!")
        return fig
    
    def plot_comprehensive_analysis(self, points, valid_indices, normals, 
                                   planarity, curvature, full_shape):
        """Create multi-panel comprehensive visualization."""
        print(f"📊 Creating comprehensive analysis plots...")
        
        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Height Map', 'Curvature Map', 
                          'Planarity Map', 'Normal Z-Component'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        # Prepare grids
        height_grid = np.full(full_shape, np.nan)
        curv_grid = np.full(full_shape, np.nan)
        plan_grid = np.full(full_shape, np.nan)
        norm_z_grid = np.full(full_shape, np.nan)
        
        for i, (idx, pt) in enumerate(zip(valid_indices, points)):
            height_grid[idx[0], idx[1]] = pt[2]
            curv_grid[idx[0], idx[1]] = curvature[i]
            plan_grid[idx[0], idx[1]] = planarity[i]
            norm_z_grid[idx[0], idx[1]] = normals[i, 2]
        
        # Add heatmaps
        fig.add_trace(go.Heatmap(z=height_grid, colorscale='Viridis', 
                                 colorbar=dict(x=0.46, len=0.4)),
                     row=1, col=1)
        fig.add_trace(go.Heatmap(z=curv_grid, colorscale='Hot',
                                 colorbar=dict(x=1.02, len=0.4)),
                     row=1, col=2)
        fig.add_trace(go.Heatmap(z=plan_grid, colorscale='RdYlGn',
                                 colorbar=dict(x=0.46, y=0.18, len=0.4)),
                     row=2, col=1)
        fig.add_trace(go.Heatmap(z=norm_z_grid, colorscale='Blues',
                                 colorbar=dict(x=1.02, y=0.18, len=0.4)),
                     row=2, col=2)
        
        fig.update_layout(
            title_text="Comprehensive Surface Analysis",
            width=1400,
            height=1000,
            showlegend=False
        )
        
        fig.show()
        print(f"✅ Comprehensive plots created!")
        return fig
    
    def analyze_file(self, filepath, k_neighbors=20, subsample=2, 
                    plot_3d=True, plot_heatmaps=True, plot_histograms=True,
                    plot_comprehensive=True):
        """
        Complete analysis pipeline with Open3D and Plotly visualization.
        
        Args:
            filepath: Path to Keyence CSV file
            k_neighbors: Number of neighbors for PCA (10-50 typical)
            subsample: Point reduction factor (1=all, 2=half, 4=quarter)
            plot_3d: Create 3D surface plots
            plot_heatmaps: Create 2D heatmaps
            plot_histograms: Create distribution histograms
            plot_comprehensive: Create multi-panel overview
            
        Returns:
            dict: Complete analysis results
        """
        print("=" * 70)
        print("🔍 KEYENCE SURFACE ANALYSIS WITH OPEN3D + PLOTLY")
        print("=" * 70)
        
        # Load data
        height_data, filename = self.load_data(filepath)
        
        # Convert to point cloud
        pcd, points, valid_indices = self.data_to_point_cloud(height_data, subsample)
        
        # Compute properties with Open3D
        properties = self.compute_normals_and_properties(pcd, points, k_neighbors)
        
        normals = properties['normals']
        planarity = properties['planarity']
        curvature = properties['curvature']
        
        # Compute statistics
        stats = self.compute_surface_statistics(points, normals, planarity, curvature)
        
        # Visualizations
        print(f"\n📊 CREATING VISUALIZATIONS")
        print("=" * 60)
        
        if plot_comprehensive:
            self.plot_comprehensive_analysis(points, valid_indices, normals,
                                            planarity, curvature, height_data.shape)
        
        if plot_3d:
            self.plot_3d_surface_with_properties(points, valid_indices, curvature,
                                                "Curvature", height_data.shape, 
                                                colorscale='Hot')
            
            self.plot_3d_surface_with_properties(points, valid_indices, planarity,
                                                "Planarity", height_data.shape,
                                                colorscale='RdYlGn')
        
        if plot_heatmaps:
            self.plot_2d_heatmap(valid_indices, curvature, "Curvature",
                               height_data.shape, colorscale='Hot')
            
            self.plot_2d_heatmap(valid_indices, planarity, "Planarity",
                               height_data.shape, colorscale='RdYlGn')
        
        if plot_histograms:
            self.plot_property_histogram(curvature, "Curvature")
            self.plot_property_histogram(planarity, "Planarity")
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 70)
        
        return {
            'filename': filename,
            'pcd': pcd,
            'points': points,
            'valid_indices': valid_indices,
            'normals': normals,
            'planarity': planarity,
            'curvature': curvature,
            'eigenvalues': properties['eigenvalues'],
            'statistics': stats,
            'full_shape': height_data.shape
        }
    
    def analyze_all_files(self, folder_path="KEYENCE_DATASET", k_neighbors=20, 
                         subsample=2, plot_3d=True, plot_heatmaps=True, 
                         plot_histograms=True, plot_comprehensive=True):
        """
        Batch process all CSV files in a folder with Open3D analysis.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            k_neighbors: Number of neighbors for PCA (10-50 typical)
            subsample: Point reduction factor (1=all, 2=half, 4=quarter)
            plot_3d: Create 3D surface plots
            plot_heatmaps: Create 2D heatmaps
            plot_histograms: Create distribution histograms
            plot_comprehensive: Create multi-panel overview
            
        Returns:
            dict: Results dictionary with analysis for each file
        """
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {folder_path}")
            return {}
        
        print(f"🔍 BATCH OPEN3D ANALYSIS")
        print(f"Found {len(csv_files)} CSV files")
        print("=" * 70)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*70}")
            print(f"ANALYZING FILE {i} OF {len(csv_files)}")
            print(f"{'='*70}")
            
            try:
                result = self.analyze_file(
                    csv_file, 
                    k_neighbors=k_neighbors,
                    subsample=subsample,
                    plot_3d=plot_3d,
                    plot_heatmaps=plot_heatmaps,
                    plot_histograms=plot_histograms,
                    plot_comprehensive=plot_comprehensive
                )
                
                self.results[result['filename']] = {
                    'status': 'success',
                    'statistics': result['statistics'],
                    'num_points': len(result['points'])
                }
                
            except Exception as e:
                filename = os.path.basename(csv_file)
                print(f"❌ Error analyzing {filename}: {e}")
                self.results[filename] = {'status': 'failed', 'error': str(e)}
            
            if i < len(csv_files):
                input(f"\n⏸️  Press Enter to continue to next file ({i+1}/{len(csv_files)})...")
        
        print(f"\n🎉 BATCH ANALYSIS COMPLETE!")
        print(f"Successfully analyzed: {sum(1 for r in self.results.values() if r.get('status') == 'success')} files")
        print(f"Failed analyses: {sum(1 for r in self.results.values() if r.get('status') == 'failed')} files")
        
        return self.results


# Convenience functions
def analyze_keyence(filepath, k_neighbors=20, subsample=2, **plot_options):
    """
    Quick analysis function for a single file.
    
    Args:
        filepath: Path to CSV file
        k_neighbors: Neighbors for PCA (default 20)
        subsample: Point reduction (default 2 = half points)
        **plot_options: plot_3d, plot_heatmaps, plot_histograms, plot_comprehensive
        
    Example:
        analyzer, results = analyze_keyence('surface.csv')
        analyzer, results = analyze_keyence('surface.csv', k_neighbors=30, subsample=1)
    """
    analyzer = KeyenceAnalyzerOpen3D()
    results = analyzer.analyze_file(filepath, k_neighbors, subsample, **plot_options)
    return analyzer, results


def quick_keyence_analysis(folder_path, k_neighbors=20, 
                           subsample=2, **plot_options):
    """
    Analyze all Keyence CSV files in a folder with Open3D analysis.
    
    Args:
        folder_path (str): Path to folder containing CSV files (default: "KEYENCE_DATASET")
        k_neighbors: Number of neighbors for PCA (default 20)
        subsample: Point reduction factor (default 2 = half points)
        **plot_options: plot_3d, plot_heatmaps, plot_histograms, plot_comprehensive
        
    Returns:
        tuple: (analyzer, results)
        
    Example:
        analyzer, results = quick_keyence_analysis("KEYENCE_DATASET", k_neighbors=30)
        analyzer, results = quick_keyence_analysis("my_data", subsample=4, plot_3d=False)
    """
    analyzer = KeyenceAnalyzerOpen3D()
    results = analyzer.analyze_all_files(
        folder_path, 
        k_neighbors=k_neighbors, 
        subsample=subsample, 
        **plot_options
    )
    return analyzer, results


if __name__ == "__main__":
    print("Keyence Analyzer with Open3D + Plotly")
    print("=" * 60)
    print("\nRequirements:")
    print("  - Python 3.7-3.11 (NOT 3.12 or 3.13)")
    print("  - open3d, numpy, pandas, plotly")
    print("\nUsage:")
    print("  # Single file analysis:")
    print("  analyzer, results = analyze_keyence('file.csv')")
    print("  analyzer, results = analyze_keyence('file.csv', k_neighbors=30, subsample=1)")
    print("\n  # Batch analysis (all CSV files in folder):")
    print("  analyzer, results = quick_keyence_analysis()")
    print("  analyzer, results = quick_keyence_analysis('KEYENCE_DATASET')")
    print("  analyzer, results = quick_keyence_analysis('my_data', k_neighbors=30, subsample=4)")
    print("\nFeatures:")
    print("  - Fast Open3D normal computation")
    print("  - Interactive Plotly visualizations")
    print("  - Curvature and planarity analysis")
    print("  - Comprehensive multi-panel plots")
    print("  - Batch processing of multiple files")