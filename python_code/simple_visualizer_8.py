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
    
    def analyze_single_file(self, filepath, display_sample_size=500):
        """
        Analyze a single Keyence CSV file and create 3D surface plot.
        
        Loads full data into memory but samples for display to improve performance.
        Invalid points (-99999.9999) are converted to NaN and excluded from plot.
        
        Args:
            filepath (str): Path to CSV file
            display_sample_size (int): Max rows/cols to display (default 500 for ~250k points)
            
        Returns:
            tuple: (full_data, display_data)
                - full_data: Complete 2D array with all data points
                - display_data: Sampled 2D array for visualization
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
            
            # Create display sample
            max_rows = min(display_sample_size, full_data.shape[0])
            max_cols = min(display_sample_size, full_data.shape[1])
            
            row_indices = np.linspace(0, full_data.shape[0]-1, max_rows, dtype=int)
            col_indices = np.linspace(0, full_data.shape[1]-1, max_cols, dtype=int)
            
            display_data = full_data[np.ix_(row_indices, col_indices)]
            
            print(f"📊 Display sample: {display_data.shape[0]} × {display_data.shape[1]} = {display_data.size:,} points")
            print(f"   (Sampled from full dataset for performance)")
            
            # Calculate physical dimensions
            physical_x = full_data.shape[1] * self.pixel_pitch_x  # μm
            physical_y = full_data.shape[0] * self.pixel_pitch_y  # μm
            print(f"📏 Physical dimensions: {physical_x:.1f} × {physical_y:.1f} μm")
            
            # Create 3D surface plot
            self._create_3d_surface(display_data, filename, row_indices, col_indices)
            
            return full_data, display_data
            
        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}")
            return None, None
    
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
    
    def analyze_all_files(self, folder_path="KEYENCE_DATASET", display_sample_size=500):
        """
        Batch process all CSV files in a folder.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            display_sample_size (int): Max rows/cols to display for each file
            
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
            
            full_data, display_data = self.analyze_single_file(csv_file, display_sample_size)
            
            if full_data is not None:
                self.results[os.path.basename(csv_file)] = {
                    'status': 'success',
                    'shape': full_data.shape,
                    'valid_points': np.sum(~np.isnan(full_data))
                }
            else:
                self.results[os.path.basename(csv_file)] = {'status': 'failed'}
            
            if i < len(csv_files):
                input(f"\n⏸️  Press Enter to continue to next file ({i+1}/{len(csv_files)})...")
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"Successfully analyzed: {sum(1 for r in self.results.values() if r.get('status') == 'success')} files")
        print(f"Failed analyses: {sum(1 for r in self.results.values() if r.get('status') == 'failed')} files")
        
        return self.results


# Convenience functions
def analyze_single_keyence_file(filepath, display_sample_size=500):
    """
    Analyze a single Keyence CSV file with 3D visualization.
    
    Args:
        filepath (str): Path to CSV file
        display_sample_size (int): Max rows/cols for display (default 500)
        
    Returns:
        tuple: (analyzer, full_data, display_data)
    """
    analyzer = SimplifiedKeyenceAnalyzer()
    full_data, display_data = analyzer.analyze_single_file(filepath, display_sample_size)
    return analyzer, full_data, display_data


def analyze_all_keyence_files(folder_path="KEYENCE_DATASET", display_sample_size=500):
    """
    Analyze all Keyence CSV files in a folder with 3D plots.
    
    Args:
        folder_path (str): Path to folder containing CSV files
        display_sample_size (int): Max rows/cols for display
        
    Returns:
        tuple: (analyzer, results)
    """
    analyzer = SimplifiedKeyenceAnalyzer()
    results = analyzer.analyze_all_files(folder_path, display_sample_size)
    return analyzer, results


def quick_keyence_analysis(filepath_or_folder, display_sample_size=500):
    """
    Smart analysis - auto-detects file or folder.
    
    Args:
        filepath_or_folder (str): Path to CSV file or folder
        display_sample_size (int): Max rows/cols for display
    """
    if os.path.isfile(filepath_or_folder) and filepath_or_folder.endswith('.csv'):
        print("🔍 Single file detected")
        return analyze_single_keyence_file(filepath_or_folder, display_sample_size)
    elif os.path.isdir(filepath_or_folder):
        print(f"📁 Folder detected - analyzing all CSV files")
        return analyze_all_keyence_files(filepath_or_folder, display_sample_size)
    else:
        print(f"❌ Invalid path: {filepath_or_folder}")
        return None


if __name__ == "__main__":
    print("Simplified Keyence Analyzer - 3D Surface Only")
    print("=" * 50)
    print("\nFeatures:")
    print("  - Loads full dataset (all points)")
    print("  - Samples for display (adjustable, default 500×500)")
    print("  - Physical coordinates (2.5 μm pixel pitch)")
    print("  - Invalid points excluded from visualization")
    print("\nUsage:")
    print("  analyzer, full, display = analyze_single_keyence_file('file.csv')")
    print("  analyzer, full, display = analyze_single_keyence_file('file.csv', display_sample_size=1000)")
    print("  analyzer, results = analyze_all_keyence_files('KEYENCE_DATASET')")
    print("  result = quick_keyence_analysis('KEYENCE_DATASET')")