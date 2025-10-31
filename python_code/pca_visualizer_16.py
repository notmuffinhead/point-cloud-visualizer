import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from multiprocessing import Pool, cpu_count
import time

class PCAKeyenceAnalyzer:
    """
    PCA-based analyzer for Keyence profilometer CSV data files.
    Uses local PCA plane fitting to remove spikes while preserving steep slopes.
    
    Keyence specifications:
    - X pixel pitch: 2.5 μm
    - Y pixel pitch: 2.5 μm (nominal, may be slightly distorted due to travel)
    - Z reference: center of sensor FOV
    - X/Y origin: top left corner of image
    - Invalid data marker: -99999.9999
    
    Method:
    - Divides surface into square tiles
    - Fits plane to each tile using PCA
    - Measures perpendicular distance from fitted plane
    - Removes points beyond threshold × std_dev
    - Parallel processing for speed
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.invalid_value = -99999.9999
        self.pixel_pitch_x = 2.5  # μm
        self.pixel_pitch_y = 2.5  # μm
        self.results = {}
    
    @staticmethod
    def _fit_plane_pca(points_3d):
        """
        Fit a plane to 3D points using PCA.
        
        Args:
            points_3d: Nx3 array of (x, y, z) coordinates
            
        Returns:
            tuple: (centroid, normal) where normal is the plane's normal vector
        """
        # Center the points
        centroid = np.mean(points_3d, axis=0)
        centered = points_3d - centroid
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Normal is eigenvector with smallest eigenvalue (least variation)
        normal = eigenvectors[:, 0]
        
        return centroid, normal
    
    @staticmethod
    def _process_tile(args):
        """
        Process a single tile (designed for multiprocessing).
        
        Args:
            args: tuple of (tile_data, row_start, row_end, col_start, col_end, 
                           pixel_pitch_x, pixel_pitch_y, std_threshold)
                           
        Returns:
            tuple: (tile_spike_mask, tile_deviations, tile_stats)
        """
        (Z_um, row_start, row_end, col_start, col_end, 
         pixel_pitch_x, pixel_pitch_y, std_threshold) = args
        
        # Extract tile
        tile_z = Z_um[row_start:row_end, col_start:col_end]
        tile_rows, tile_cols = tile_z.shape
        
        # Initialize outputs
        tile_spike_mask = np.zeros((tile_rows, tile_cols), dtype=bool)
        tile_deviations = np.full((tile_rows, tile_cols), np.nan)
        
        # Find valid points
        valid_mask = ~np.isnan(tile_z)
        num_valid = np.sum(valid_mask)
        
        # Need at least 3 points to fit a plane
        if num_valid < 3:
            return tile_spike_mask, tile_deviations, {'num_valid': 0, 'num_spikes': 0}
        
        # Create 3D coordinates for valid points
        rows_idx, cols_idx = np.where(valid_mask)
        
        # Physical coordinates
        x_coords = (col_start + cols_idx) * pixel_pitch_x
        y_coords = (row_start + rows_idx) * pixel_pitch_y
        z_coords = tile_z[valid_mask]
        
        points_3d = np.column_stack([x_coords, y_coords, z_coords])
        
        # Fit plane using PCA
        centroid, normal = PCAKeyenceAnalyzer._fit_plane_pca(points_3d)
        
        # Calculate perpendicular distance from each point to the plane
        # Distance = |dot(point - centroid, normal)|
        deviations = np.abs(np.dot(points_3d - centroid, normal))
        
        # Store deviations back in tile array
        tile_deviations[valid_mask] = deviations
        
        # Calculate statistics on deviations
        mean_dev = np.mean(deviations)
        std_dev = np.std(deviations)
        
        # Mark spikes (beyond threshold × std_dev from mean)
        spike_threshold = mean_dev + std_threshold * std_dev
        spike_indices = deviations > spike_threshold
        
        # Map spike indices back to tile coordinates
        spike_rows = rows_idx[spike_indices]
        spike_cols = cols_idx[spike_indices]
        tile_spike_mask[spike_rows, spike_cols] = True
        
        num_spikes = np.sum(spike_indices)
        
        stats = {
            'num_valid': num_valid,
            'num_spikes': num_spikes,
            'mean_dev': mean_dev,
            'std_dev': std_dev,
            'max_dev': np.max(deviations)
        }
        
        return tile_spike_mask, tile_deviations, stats
    
    def detect_spikes_pca(self, full_data, tile_size=100, std_threshold=3.0, n_jobs=-1):
        """
        PCA-based spike detection with parallel processing.
        
        Args:
            full_data: 2D numpy array of height values (mm)
            tile_size: Size of square tiles in pixels (default 100)
            std_threshold: Number of std_devs beyond which to remove points (default 3.0)
            n_jobs: Number of parallel jobs (-1 = all cores, default -1)
            
        Returns:
            dict: Analysis results including spike mask, cleaned data, and statistics
        """
        print(f"\n🔬 PCA-BASED SPIKE DETECTION")
        print("=" * 60)
        print(f"📊 Method: Local PCA plane fitting with parallel processing")
        print(f"   Tile size: {tile_size}×{tile_size} pixels ({tile_size*self.pixel_pitch_x:.1f} μm)")
        print(f"   Std threshold: {std_threshold}σ (removes points > {std_threshold}× std_dev from plane)")
        
        rows, cols = full_data.shape
        print(f"   Data size: {rows} × {cols} = {full_data.size:,} points")
        
        # Determine number of cores
        if n_jobs == -1:
            n_jobs = cpu_count()
        print(f"   CPU cores: {n_jobs}")
        
        # Calculate number of tiles
        num_tiles_y = (rows + tile_size - 1) // tile_size
        num_tiles_x = (cols + tile_size - 1) // tile_size
        total_tiles = num_tiles_y * num_tiles_x
        print(f"   Grid: {num_tiles_y} × {num_tiles_x} = {total_tiles} tiles")
        
        # Convert heights from mm to μm
        Z_um = full_data * 1000
        
        # Prepare tile arguments for parallel processing
        tile_args = []
        tile_positions = []
        
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                row_start = tile_y * tile_size
                row_end = min((tile_y + 1) * tile_size, rows)
                col_start = tile_x * tile_size
                col_end = min((tile_x + 1) * tile_size, cols)
                
                tile_args.append((
                    Z_um, row_start, row_end, col_start, col_end,
                    self.pixel_pitch_x, self.pixel_pitch_y, std_threshold
                ))
                tile_positions.append((row_start, row_end, col_start, col_end))
        
        print(f"\n🚀 Processing tiles in parallel...")
        start_time = time.time()
        
        # Process tiles in parallel
        with Pool(processes=n_jobs) as pool:
            results = pool.map(self._process_tile, tile_args)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Processing complete in {elapsed:.2f} seconds")
        print(f"   Throughput: {total_tiles/elapsed:.1f} tiles/sec")
        
        # Reconstruct full arrays from tile results
        spike_mask = np.zeros(full_data.shape, dtype=bool)
        deviation_map = np.full(full_data.shape, np.nan)
        
        all_stats = []
        for (row_start, row_end, col_start, col_end), (tile_mask, tile_devs, stats) in zip(tile_positions, results):
            spike_mask[row_start:row_end, col_start:col_end] = tile_mask
            deviation_map[row_start:row_end, col_start:col_end] = tile_devs
            if stats['num_valid'] > 0:
                all_stats.append(stats)
        
        # Calculate global statistics
        num_spikes = np.sum(spike_mask)
        num_valid = np.sum(~np.isnan(Z_um))
        tiles_with_spikes = sum(1 for s in all_stats if s['num_spikes'] > 0)
        
        print(f"\n📊 Tile statistics:")
        if all_stats:
            mean_devs = [s['mean_dev'] for s in all_stats]
            std_devs = [s['std_dev'] for s in all_stats]
            max_devs = [s['max_dev'] for s in all_stats]
            
            print(f"   Mean deviation across tiles: {np.mean(mean_devs):.2f} μm")
            print(f"   Mean std_dev across tiles: {np.mean(std_devs):.2f} μm")
            print(f"   Max deviation found: {np.max(max_devs):.2f} μm")
        
        print(f"\n🎯 Spike detection results:")
        print(f"   Tiles with spikes: {tiles_with_spikes}/{total_tiles}")
        print(f"   Spikes detected: {num_spikes:,} / {num_valid:,} "
              f"({100*num_spikes/num_valid:.2f}%)")
        
        # Create cleaned data
        cleaned_data = full_data.copy()
        cleaned_data[spike_mask] = np.nan
        
        return {
            'spike_mask': spike_mask,
            'cleaned_data': cleaned_data,
            'deviation_map': deviation_map,
            'num_spikes': num_spikes,
            'num_valid': num_valid,
            'num_tiles': total_tiles,
            'tiles_with_spikes': tiles_with_spikes,
            'processing_time': elapsed,
            'tile_stats': all_stats
        }
    
    def analyze_single_file(self, filepath, display_sample_size=500, 
                          tile_size=100, std_threshold=3.0, n_jobs=-1):
        """
        Analyze a single Keyence CSV file with PCA-based spike removal.
        
        Args:
            filepath (str): Path to CSV file
            display_sample_size (int): Max rows/cols to display (default 500)
            tile_size (int): Size of square tiles in pixels (default 100)
            std_threshold (float): Std dev multiplier threshold (default 3.0)
            n_jobs (int): Number of parallel jobs (-1 = all cores)
            
        Returns:
            tuple: (full_data, display_data, pca_results)
        """
        filename = os.path.basename(filepath)
        print("🔬 KEYENCE 3D SURFACE ANALYSIS - PCA METHOD")
        print("=" * 60)
        print(f"📁 File: {filename}")
        
        try:
            # Load the FULL CSV file
            full_df = pd.read_csv(filepath, header=None)
            print(f"📋 Full dataset: {full_df.shape[0]} rows × {full_df.shape[1]} columns")
            print(f"   Total points: {full_df.size:,}")
            
            # Convert to NumPy array
            full_data = full_df.values.copy()
            
            # Replace invalid values with NaN
            invalid_mask = full_data == self.invalid_value
            full_data[invalid_mask] = np.nan
            
            # Count invalid points
            num_invalid = np.sum(invalid_mask)
            total_points = full_data.size
            print(f"📊 Invalid points: {num_invalid:,}/{total_points:,} ({100*num_invalid/total_points:.1f}%)")
            
            # Calculate physical dimensions
            physical_x = full_data.shape[1] * self.pixel_pitch_x  # μm
            physical_y = full_data.shape[0] * self.pixel_pitch_y  # μm
            print(f"📏 Physical dimensions: {physical_x:.1f} × {physical_y:.1f} μm")
            
            # Apply PCA-based spike detection
            pca_results = self.detect_spikes_pca(full_data, tile_size, std_threshold, n_jobs)
            data_to_display = pca_results['cleaned_data']
            
            print(f"\n✅ PCA-based spike detection complete - spikes removed")
            
            # Create display sample
            max_rows = min(display_sample_size, data_to_display.shape[0])
            max_cols = min(display_sample_size, data_to_display.shape[1])
            
            row_indices = np.linspace(0, data_to_display.shape[0]-1, max_rows, dtype=int)
            col_indices = np.linspace(0, data_to_display.shape[1]-1, max_cols, dtype=int)
            
            display_data = data_to_display[np.ix_(row_indices, col_indices)]
            deviation_display = pca_results['deviation_map'][np.ix_(row_indices, col_indices)]
            
            print(f"\n📊 Display sample: {display_data.shape[0]} × {display_data.shape[1]} = {display_data.size:,} points")
            print(f"   (Sampled from full dataset for performance)")
            
            # Create 3D surface plot
            self._create_3d_surface(display_data, filename, row_indices, col_indices)
            
            # Create deviation heatmap
            self._create_deviation_heatmap(deviation_display, filename, row_indices, col_indices)
            
            return full_data, display_data, pca_results
            
        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _create_3d_surface(self, heights, filename, row_indices, col_indices):
        """
        Create 3D surface topography visualization with physical coordinates.
        
        Args:
            heights: NumPy 2D array of surface heights in mm (sampled)
            filename: Name of file (used in plot title)
            row_indices: Original row indices used for sampling
            col_indices: Original column indices used for sampling
        """
        print(f"\n📊 Creating 3D surface visualization...")
        
        # Create coordinate grids in physical units (μm)
        X_pixels = col_indices
        Y_pixels = row_indices
        X_um = X_pixels * self.pixel_pitch_x  # Convert to μm
        Y_um = Y_pixels * self.pixel_pitch_y  # Convert to μm
        
        X, Y = np.meshgrid(X_um, Y_um)
        
        # Convert heights from mm to μm
        Z_um = heights * 1000
        
        # Create figure
        fig = go.Figure(data=[
            go.Surface(
                x=X, 
                y=Y, 
                z=Z_um, 
                colorscale='Viridis',
                colorbar=dict(title="Height (μm)")
            )
        ])
        
        fig.update_layout(
            title=f"3D Surface Topography: {filename} (PCA Cleaned)",
            scene=dict(
                xaxis_title="X Position (μm)",
                yaxis_title="Y Position (μm)",
                zaxis_title="Height (μm)",
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        fig.show()
        print("   ✅ 3D surface plot created successfully!")
        return fig
    
    def _create_deviation_heatmap(self, deviations, filename, row_indices, col_indices):
        """
        Create heatmap showing perpendicular deviation from fitted planes.
        
        Args:
            deviations: NumPy 2D array of deviations in μm (sampled)
            filename: Name of file (used in plot title)
            row_indices: Original row indices used for sampling
            col_indices: Original column indices used for sampling
        """
        print(f"📊 Creating deviation heatmap...")
        
        # Create coordinate grids in physical units (μm)
        X_pixels = col_indices
        Y_pixels = row_indices
        X_um = X_pixels * self.pixel_pitch_x
        Y_um = Y_pixels * self.pixel_pitch_y
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=X_um,
            y=Y_um,
            z=deviations,
            colorscale='Hot',
            colorbar=dict(title="Deviation (μm)"),
            hovertemplate='X: %{x:.1f} μm<br>Y: %{y:.1f} μm<br>Deviation: %{z:.2f} μm<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"PCA Deviation Map: {filename}",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=1000,
            height=800
        )
        
        fig.show()
        print("   ✅ Deviation heatmap created successfully!")
        return fig
    
    def analyze_all_files(self, folder_path="KEYENCE_DATASET", display_sample_size=500,
                         tile_size=100, std_threshold=3.0, n_jobs=-1):
        """
        Batch process all CSV files in a folder.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            display_sample_size (int): Max rows/cols to display for each file
            tile_size (int): Size of square tiles in pixels
            std_threshold (float): Std dev multiplier threshold
            n_jobs (int): Number of parallel jobs
            
        Returns:
            dict: Results dictionary
        """
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {folder_path}")
            return
        
        print(f"🔬 PCA ANALYSIS OF ALL FILES")
        print(f"Found {len(csv_files)} CSV files")
        print("=" * 60)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*60}")
            print(f"ANALYZING FILE {i} OF {len(csv_files)}")
            print(f"{'='*60}")
            
            full_data, display_data, pca_results = self.analyze_single_file(
                csv_file, display_sample_size, tile_size, std_threshold, n_jobs
            )
            
            if full_data is not None:
                result_entry = {
                    'status': 'success',
                    'shape': full_data.shape,
                    'valid_points': np.sum(~np.isnan(full_data))
                }
                if pca_results:
                    result_entry['pca_results'] = pca_results
                self.results[os.path.basename(csv_file)] = result_entry
            else:
                self.results[os.path.basename(csv_file)] = {'status': 'failed'}
            
            if i < len(csv_files):
                input(f"\n⏸️  Press Enter to continue to next file ({i+1}/{len(csv_files)})...")
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"Successfully analyzed: {sum(1 for r in self.results.values() if r.get('status') == 'success')} files")
        print(f"Failed analyses: {sum(1 for r in self.results.values() if r.get('status') == 'failed')} files")
        
        return self.results


# Convenience functions
def analyze_single_keyence_file(filepath, display_sample_size=500, 
                               tile_size=100, std_threshold=3.0, n_jobs=-1):
    """
    Analyze a single Keyence CSV file with PCA-based spike removal.
    
    Args:
        filepath (str): Path to CSV file
        display_sample_size (int): Max rows/cols for display (default 500)
        tile_size (int): Square tile size in pixels (default 100 = 250×250 μm)
        std_threshold (float): Std dev multiplier (default 3.0 = remove if >3σ from plane)
        n_jobs (int): Number of CPU cores to use (-1 = all cores)
        
    Returns:
        tuple: (analyzer, full_data, display_data, pca_results)
    """
    analyzer = PCAKeyenceAnalyzer()
    full_data, display_data, pca_results = analyzer.analyze_single_file(
        filepath, display_sample_size, tile_size, std_threshold, n_jobs
    )
    return analyzer, full_data, display_data, pca_results


def analyze_all_keyence_files(folder_path="KEYENCE_DATASET", display_sample_size=500,
                              tile_size=100, std_threshold=3.0, n_jobs=-1):
    """
    Analyze all Keyence CSV files in a folder with PCA-based spike removal.
    
    Args:
        folder_path (str): Path to folder containing CSV files
        display_sample_size (int): Max rows/cols for display
        tile_size (int): Square tile size in pixels
        std_threshold (float): Std dev multiplier threshold
        n_jobs (int): Number of CPU cores to use
        
    Returns:
        tuple: (analyzer, results)
    """
    analyzer = PCAKeyenceAnalyzer()
    results = analyzer.analyze_all_files(folder_path, display_sample_size, 
                                        tile_size, std_threshold, n_jobs)
    return analyzer, results


def quick_keyence_analysis(filepath_or_folder, display_sample_size=500,
                          tile_size=100, std_threshold=3.0, n_jobs=-1):
    """
    Smart analysis - auto-detects file or folder.
    
    Args:
        filepath_or_folder (str): Path to CSV file or folder
        display_sample_size (int): Max rows/cols for display
        tile_size (int): Square tile size in pixels
        std_threshold (float): Std dev multiplier threshold
        n_jobs (int): Number of CPU cores to use
    """
    if os.path.isfile(filepath_or_folder) and filepath_or_folder.endswith('.csv'):
        print("🔬 Single file detected")
        return analyze_single_keyence_file(filepath_or_folder, display_sample_size,
                                          tile_size, std_threshold, n_jobs)
    elif os.path.isdir(filepath_or_folder):
        print(f"🔬 Folder detected - analyzing all CSV files")
        return analyze_all_keyence_files(filepath_or_folder, display_sample_size,
                                        tile_size, std_threshold, n_jobs)
    else:
        print(f"❌ Invalid path: {filepath_or_folder}")
        return None


if __name__ == "__main__":
    print("Keyence Analyzer with PCA-Based Spike Detection")
    print("=" * 50)
    print("\nFeatures:")
    print("  ✅ Local PCA plane fitting (geometry-aware)")
    print("  ✅ Preserves steep slopes (fits to local surface)")
    print("  ✅ Parallel processing (uses all CPU cores)")
    print("  ✅ Fast performance (5-10 sec for ~12M points)")
    print("  ✅ Deviation heatmap visualization")
    print("\nHow it works:")
    print("  1. Divide surface into square tiles")
    print("  2. Fit plane to each tile using PCA")
    print("  3. Calculate perpendicular distance from plane")
    print("  4. Remove points > threshold × std_dev")
    print("  5. Process tiles in parallel for speed")
    print("\nDefault parameters:")
    print("  - tile_size: 100 pixels (250×250 μm)")
    print("  - std_threshold: 3.0 (removes points >3σ from fitted plane)")
    print("  - n_jobs: -1 (uses all CPU cores)")
    print("\nUsage Examples:")
    print("  IMPORTANT: On Windows, always wrap your code in 'if __name__ == \"__main__\":'")
    print("  ")
    print("  # Example 1: Single file")
    print("  if __name__ == '__main__':")
    print("      analyzer, full, display, pca = analyze_single_keyence_file('file.csv')")
    print("  ")
    print("  # Example 2: Adjust parameters")
    print("  if __name__ == '__main__':")
    print("      analyzer, full, display, pca = analyze_single_keyence_file(")
    print("          'file.csv', tile_size=50, std_threshold=5.0)")
    print("  ")
    print("  # Example 3: Batch process folder")
    print("  if __name__ == '__main__':")
    print("      analyzer, results = analyze_all_keyence_files('KEYENCE_DATASET')")
    print("  ")
    print("  # Example 4: Quick analysis")
    print("  if __name__ == '__main__':")
    print("      result = quick_keyence_analysis('KEYENCE_DATASET')")
    print("\n" + "=" * 50)
    print("To run: Uncomment one of the examples below and execute this script")
    print("=" * 50 + "\n")
    
    # ========== UNCOMMENT ONE OF THESE TO RUN ==========
    
    # Example 1: Analyze single file
    # analyzer, full, display, pca = analyze_single_keyence_file('your_file.csv')
    
    # Example 2: Analyze all files in folder
    # analyzer, results = analyze_all_keyence_files('KEYENCE_DATASET')
    
    # Example 3: Quick analysis (auto-detects file or folder)
    # result = quick_keyence_analysis('KEYENCE_DATASET')
    
    # Example 4: Custom parameters
    # analyzer, full, display, pca = analyze_single_keyence_file(
    #     'your_file.csv',
    #     tile_size=50,        # Smaller tiles for more local fitting
    #     std_threshold=5.0,   # More lenient threshold
    #     n_jobs=4             # Use 4 CPU cores
    # )