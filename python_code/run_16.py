"""
Windows-Safe Example Script for PCA Keyence Analyzer

IMPORTANT: On Windows, multiprocessing requires the if __name__ == '__main__' guard.
This script shows the correct way to use the analyzer.

Copy this file and modify the paths for your data!
"""

from pca_visualizer_16 import (
    analyze_single_keyence_file,
    analyze_all_keyence_files,
    quick_keyence_analysis
)

def main():
    """Main function with all your analysis code."""
    
    # ========== CONFIGURATION ==========
    # Modify these paths for your data
    single_file_path = '250825_095513_height.csv'  # Change to your file
    folder_path = 'KEYENCE_DATASET_PYTHON_COPY'                 # Change to your folder
    
    # Parameters (adjust as needed)
    tile_size = 100        # pixels (100 = 250×250 μm tiles)
    std_threshold = 3.0    # remove points > 3σ from fitted plane
    n_jobs = -1            # -1 = use all CPU cores
    
    # ========== CHOOSE YOUR ANALYSIS MODE ==========
    
    # MODE 1: Analyze a single file
    # print("Analyzing single file...")
    # analyzer, full_data, display_data, pca_results = analyze_single_keyence_file(
    #     filepath=single_file_path,
    #     tile_size=tile_size,
    #     std_threshold=std_threshold,
    #     n_jobs=n_jobs
    # )
    
    # MODE 2: Analyze all files in a folder
    print("Analyzing all files in folder...")
    analyzer, results = analyze_all_keyence_files(
        folder_path="KEYENCE_DATASET",
        tile_size=tile_size,
        std_threshold=std_threshold,
        n_jobs=n_jobs
    )
    
    # MODE 3: Quick analysis (auto-detects file or folder)
    # print("Quick analysis...")
    # result = quick_keyence_analysis(
    #     filepath_or_folder=folder_path,  # or single_file_path
    #     tile_size=tile_size,
    #     std_threshold=std_threshold,
    #     n_jobs=n_jobs
    # )
    
    # ========== ACCESS RESULTS ==========
    
    if pca_results:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Spikes detected: {pca_results['num_spikes']:,}")
        print(f"Processing time: {pca_results['processing_time']:.2f} seconds")
        print(f"Tiles processed: {pca_results['num_tiles']}")
        print(f"Tiles with spikes: {pca_results['tiles_with_spikes']}")
        
        # Access the cleaned data
        cleaned_data = pca_results['cleaned_data']
        print(f"\nCleaned data shape: {cleaned_data.shape}")
        
        # Access the deviation map (for further analysis if needed)
        deviation_map = pca_results['deviation_map']
        print(f"Deviation map shape: {deviation_map.shape}")


if __name__ == '__main__':
    """
    This if __name__ == '__main__' block is REQUIRED on Windows.
    
    It prevents the multiprocessing module from recursively spawning
    new processes when it imports this file.
    """
    main()