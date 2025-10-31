import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob

class SimplifiedKeyenceAnalyzer:
    """
    Simplified analyzer for Keyence profilometer CSV data files.
    Shows only 3D surface topography with invalid points excluded.
    
    Keyence specifications:
    - X pixel pitch: 2.5 μm
    - Y pixel pitch: 2.5 μm (nominal, may be slightly distorted due to travel)
    - Z reference: center of sensor FOV
    - X/Y origin: top left corner of image
    - Invalid data marker: -99999.9999
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.invalid_value = -99999.9999
        self.pixel_pitch_x = 2.5  # μm
        self.pixel_pitch_y = 2.5  # μm
        self.results = {}
    
    def detect_spikes_mad(self, full_data, tile_size=1000, mad_threshold=10):
        """
        Adaptive spike detection using MAD (Median Absolute Deviation).
        
        Strategy:
        - Divide surface into tiles
        - For each tile, calculate median height and MAD
        - Remove points where |height - median| / MAD > threshold
        
        Why this works for both flat and sloped surfaces:
        
        FLAT SURFACE:
        - MAD is small (points are tightly clustered)
        - Spike deviation / small_MAD = LARGE ratio → removed ✓
        
        SLOPED SURFACE:
        - MAD is large (natural height variation due to slope)
        - Normal point deviation / large_MAD = small ratio → kept ✓
        - Spike deviation / large_MAD = still large ratio → removed ✓
        
        The threshold is scale-invariant and adapts automatically!
        
        Args:
            full_data: 2D numpy array of height values (mm)
            tile_size: Size of grid tiles in pixels (default 100)
            mad_threshold: Number of MADs beyond which to remove points (default 5)
            
        Returns:
            dict: Analysis results including spike mask and statistics
        """
        print(f"\n🔬 ADAPTIVE MAD-BASED SPIKE DETECTION")
        print("=" * 60)
        print(f"📊 Method: Median Absolute Deviation (scale-invariant)")
        print(f"   Tile size: {tile_size}×{tile_size} pixels ({tile_size*self.pixel_pitch_x:.1f} μm)")
        print(f"   MAD threshold: {mad_threshold}× (removes points > {mad_threshold}×MAD from median)")
        
        rows, cols = full_data.shape
        print(f"   Data size: {rows} × {cols} = {full_data.size:,} points")
        
        # Calculate number of tiles
        num_tiles_y = (rows + tile_size - 1) // tile_size
        num_tiles_x = (cols + tile_size - 1) // tile_size
        total_tiles = num_tiles_y * num_tiles_x
        print(f"   Grid: {num_tiles_y} × {num_tiles_x} = {total_tiles} tiles")
        
        # Convert heights from mm to μm
        Z_um = full_data * 1000
        
        # Initialize arrays for results
        spike_mask = np.zeros(full_data.shape, dtype=bool)
        mad_ratios = np.full(full_data.shape, np.nan)  # Store the deviation ratios
        
        print(f"\n🔄 Processing tiles...")
        tiles_processed = 0
        tiles_with_spikes = 0
        
        # Track statistics across all tiles
        all_mads = []
        all_medians = []
        
        # Process each tile
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                tiles_processed += 1
                
                # Show progress every 100 tiles
                if tiles_processed % 100 == 0 or tiles_processed == total_tiles:
                    progress = 100 * tiles_processed / total_tiles
                    print(f"   Progress: {progress:.1f}% ({tiles_processed}/{total_tiles} tiles)")
                
                # Define tile boundaries
                row_start = tile_y * tile_size
                row_end = min((tile_y + 1) * tile_size, rows)
                col_start = tile_x * tile_size
                col_end = min((tile_x + 1) * tile_size, cols)
                
                # Extract tile data
                tile_z = Z_um[row_start:row_end, col_start:col_end]
                
                # Find valid points in this tile
                valid_mask = ~np.isnan(tile_z)
                num_valid = np.sum(valid_mask)
                
                # Need at least a few points to calculate statistics
                if num_valid < 3:
                    continue
                
                # Extract valid heights
                valid_heights = tile_z[valid_mask]
                
                # Calculate median (robust to outliers)
                median_height = np.median(valid_heights)
                
                # Calculate MAD (Median Absolute Deviation)
                # MAD = median of |deviations from median|
                absolute_deviations = np.abs(valid_heights - median_height)
                mad = np.median(absolute_deviations)
                
                # Store statistics
                all_mads.append(mad)
                all_medians.append(median_height)
                
                # Handle edge case where MAD = 0 (all points identical)
                if mad < 0.01:  # Essentially zero (< 0.01 μm)
                    # If MAD is zero, any deviation at all is suspicious
                    # Use a small default MAD for the ratio calculation
                    mad = 0.1  # μm
                
                # Calculate deviation ratio for all points in tile
                deviations = np.abs(tile_z - median_height)
                ratios = deviations / mad
                
                # Store ratios
                mad_ratios[row_start:row_end, col_start:col_end] = ratios
                
                # Mark spikes (only valid points can be spikes)
                tile_spikes = (ratios > mad_threshold) & valid_mask
                spike_mask[row_start:row_end, col_start:col_end] = tile_spikes
                
                if np.any(tile_spikes):
                    tiles_with_spikes += 1
        
        print(f"   ✅ Processing complete!")
        print(f"   Tiles with spikes detected: {tiles_with_spikes}/{total_tiles}")
        
        # Calculate global statistics
        num_spikes = np.sum(spike_mask)
        num_valid = np.sum(~np.isnan(Z_um))
        
        print(f"\n📊 Tile statistics:")
        print(f"   Mean MAD across tiles: {np.mean(all_mads):.2f} μm")
        print(f"   Median MAD across tiles: {np.median(all_mads):.2f} μm")
        print(f"   Min MAD: {np.min(all_mads):.2f} μm (flat regions)")
        print(f"   Max MAD: {np.max(all_mads):.2f} μm (sloped regions)")
        
        # Statistics on deviation ratios
        valid_ratios = mad_ratios[~np.isnan(mad_ratios)]
        if len(valid_ratios) > 0:
            print(f"\n📏 Deviation ratio statistics:")
            print(f"   Mean ratio: {np.mean(valid_ratios):.2f}")
            print(f"   Median ratio: {np.median(valid_ratios):.2f}")
            print(f"   95th percentile: {np.percentile(valid_ratios, 95):.2f}")
            print(f"   Max ratio: {np.max(valid_ratios):.2f}")
        
        print(f"\n🎯 Spike detection results:")
        print(f"   Spikes detected: {num_spikes:,} / {num_valid:,} "
              f"({100*num_spikes/num_valid:.2f}%)")
        
        # Create cleaned data
        cleaned_data = full_data.copy()
        cleaned_data[spike_mask] = np.nan
        
        return {
            'spike_mask': spike_mask,
            'cleaned_data': cleaned_data,
            'num_spikes': num_spikes,
            'mad_ratios': mad_ratios,
            'tile_mads': np.array(all_mads),
            'tile_medians': np.array(all_medians),
            'num_tiles': total_tiles,
            'tiles_with_spikes': tiles_with_spikes
        }
    
    def analyze_single_file(self, filepath, display_sample_size=500, 
                          apply_spike_removal=True, tile_size=1000, mad_threshold=10):
        """
        Analyze a single Keyence CSV file and create 3D surface plot.
        
        Loads full data into memory but samples for display to improve performance.
        Invalid points (-99999.9999) are converted to NaN and excluded from plot.
        
        Args:
            filepath (str): Path to CSV file
            display_sample_size (int): Max rows/cols to display (default 500)
            apply_spike_removal (bool): Whether to apply MAD-based spike detection
            tile_size (int): Size of grid tiles in pixels (default 100)
            mad_threshold (float): MAD multiplier threshold (default 5)
            
        Returns:
            tuple: (full_data, display_data, spike_results)
        """
        filename = os.path.basename(filepath)
        print("🔍 KEYENCE 3D SURFACE ANALYSIS")
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
            
            # Apply MAD-based spike detection if requested
            spike_results = None
            data_to_display = full_data
            
            if apply_spike_removal:
                spike_results = self.detect_spikes_mad(full_data, tile_size, mad_threshold)
                data_to_display = spike_results['cleaned_data']
                print(f"\n✅ MAD-based spike detection complete - spikes removed")
            
            # Create display sample
            max_rows = min(display_sample_size, data_to_display.shape[0])
            max_cols = min(display_sample_size, data_to_display.shape[1])
            
            row_indices = np.linspace(0, data_to_display.shape[0]-1, max_rows, dtype=int)
            col_indices = np.linspace(0, data_to_display.shape[1]-1, max_cols, dtype=int)
            
            display_data = data_to_display[np.ix_(row_indices, col_indices)]
            
            print(f"\n📊 Display sample: {display_data.shape[0]} × {display_data.shape[1]} = {display_data.size:,} points")
            print(f"   (Sampled from full dataset for performance)")
            
            # Calculate physical dimensions
            physical_x = full_data.shape[1] * self.pixel_pitch_x  # μm
            physical_y = full_data.shape[0] * self.pixel_pitch_y  # μm
            print(f"📏 Physical dimensions: {physical_x:.1f} × {physical_y:.1f} μm")
            
            # Create 3D surface plot
            title_suffix = " (Spikes Removed)" if apply_spike_removal else ""
            self._create_3d_surface(display_data, filename + title_suffix, 
                                   row_indices, col_indices)
            
            return full_data, display_data, spike_results
            
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
            title=f"3D Surface Topography: {filename}",
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
        print("✅ 3D surface plot created successfully!")
        return fig
    
    def analyze_all_files(self, folder_path="KEYENCE_DATASET", display_sample_size=500,
                         apply_spike_removal=True, tile_size=1000, mad_threshold=10):
        """
        Batch process all CSV files in a folder.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            display_sample_size (int): Max rows/cols to display for each file
            apply_spike_removal (bool): Whether to apply MAD-based spike detection
            tile_size (int): Size of grid tiles in pixels
            mad_threshold (float): MAD multiplier threshold
            
        Returns:
            dict: Results dictionary
        """
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {folder_path}")
            return
        
        print(f"🔍 3D ANALYSIS OF ALL FILES")
        print(f"Found {len(csv_files)} CSV files")
        print("=" * 60)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*60}")
            print(f"ANALYZING FILE {i} OF {len(csv_files)}")
            print(f"{'='*60}")
            
            full_data, display_data, spike_results = self.analyze_single_file(
                csv_file, display_sample_size, apply_spike_removal, 
                tile_size, mad_threshold
            )
            
            if full_data is not None:
                result_entry = {
                    'status': 'success',
                    'shape': full_data.shape,
                    'valid_points': np.sum(~np.isnan(full_data))
                }
                if spike_results:
                    result_entry['spike_results'] = spike_results
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
                               apply_spike_removal=True, tile_size=1000,
                               mad_threshold=10):
    """
    Analyze a single Keyence CSV file with 3D visualization and optional spike removal.
    
    Args:
        filepath (str): Path to CSV file
        display_sample_size (int): Max rows/cols for display (default 500)
        apply_spike_removal (bool): Apply MAD-based spike detection
        tile_size (int): Grid tile size in pixels (default 100)
        mad_threshold (float): MAD multiplier (default 5 = remove if >5×MAD from median)
        
    Returns:
        tuple: (analyzer, full_data, display_data, spike_results)
    """
    analyzer = SimplifiedKeyenceAnalyzer()
    full_data, display_data, spike_results = analyzer.analyze_single_file(
        filepath, display_sample_size, apply_spike_removal, 
        tile_size, mad_threshold
    )
    return analyzer, full_data, display_data, spike_results


def analyze_all_keyence_files(folder_path="KEYENCE_DATASET", display_sample_size=500,
                              apply_spike_removal=True, tile_size=1000,
                              mad_threshold=10):
    """
    Analyze all Keyence CSV files in a folder with 3D plots and optional spike removal.
    
    Args:
        folder_path (str): Path to folder containing CSV files
        display_sample_size (int): Max rows/cols for display
        apply_spike_removal (bool): Apply MAD-based spike detection
        tile_size (int): Grid tile size in pixels
        mad_threshold (float): MAD multiplier threshold
        
    Returns:
        tuple: (analyzer, results)
    """
    analyzer = SimplifiedKeyenceAnalyzer()
    results = analyzer.analyze_all_files(folder_path, display_sample_size, 
                                        apply_spike_removal, tile_size,
                                        mad_threshold)
    return analyzer, results


def quick_keyence_analysis(filepath_or_folder, display_sample_size=500,
                          apply_spike_removal=True, tile_size=1000,
                          mad_threshold=10):
    """
    Smart analysis - auto-detects file or folder.
    
    Args:
        filepath_or_folder (str): Path to CSV file or folder
        display_sample_size (int): Max rows/cols for display
        apply_spike_removal (bool): Apply MAD-based spike detection
        tile_size (int): Grid tile size in pixels
        mad_threshold (float): MAD multiplier threshold
    """
    if os.path.isfile(filepath_or_folder) and filepath_or_folder.endswith('.csv'):
        print("🔍 Single file detected")
        return analyze_single_keyence_file(filepath_or_folder, display_sample_size,
                                          apply_spike_removal, tile_size,
                                          mad_threshold)
    elif os.path.isdir(filepath_or_folder):
        print(f"📁 Folder detected - analyzing all CSV files")
        return analyze_all_keyence_files(filepath_or_folder, display_sample_size,
                                        apply_spike_removal, tile_size,
                                        mad_threshold)
    else:
        print(f"❌ Invalid path: {filepath_or_folder}")
        return None


if __name__ == "__main__":
    print("Keyence Analyzer with Adaptive MAD-Based Spike Detection")
    print("=" * 50)
    print("\nFeatures:")
    print("  - Scale-invariant spike detection using MAD")
    print("  - Automatically adapts to flat and sloped surfaces")
    print("  - Fast processing (handles millions of points)")
    print("  - No plane fitting needed!")
    print("\nHow it works:")
    print("  - Flat regions: Small MAD → strict filtering")
    print("  - Sloped regions: Large MAD → lenient filtering")
    print("  - Spikes: Always have high deviation/MAD ratio → removed")
    print("\nUsage:")
    print("  # Default settings (100×100 tiles, 5×MAD threshold)")
    print("  analyzer, full, display, spikes = analyze_single_keyence_file('file.csv')")
    print("  ")
    print("  # Adjust tile size")
    print("  analyzer, full, display, spikes = analyze_single_keyence_file('file.csv', tile_size=200)")
    print("  ")
    print("  # Adjust MAD threshold (higher = more lenient, lower = more aggressive)")
    print("  analyzer, full, display, spikes = analyze_single_keyence_file('file.csv', mad_threshold=3)")
    print("  ")
    print("  # Without spike removal")
    print("  analyzer, full, display, spikes = analyze_single_keyence_file('file.csv', apply_spike_removal=False)")
    print("  ")
    print("  # Batch process folder")
    print("  analyzer, results = analyze_all_keyence_files('KEYENCE_DATASET')")
    print("  ")
    print("  # Quick analysis")
    print("  result = quick_keyence_analysis('KEYENCE_DATASET')")