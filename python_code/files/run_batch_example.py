"""
Example script for batch processing ALL CSV files in a folder
Replace 'KEYENCE_DATASET' with your actual folder path!
"""

from keyence_batch import quick_keyence_analysis_and_clean

# CHANGE THIS to your folder path!
FOLDER_PATH = "KEYENCE_DATASET"  # <-- Put your folder path here

# Example 1: Basic batch processing (recommended)
print("Starting batch processing...")
analyzer, results = quick_keyence_analysis_and_clean(
    folder_path=FOLDER_PATH,
    subsample=4,              # Fast! Use 2 for better quality
    curvature_threshold=0.1,  # Adjust based on your data
    k_neighbors=20,
    n_jobs=-1,                # Use all CPU cores
    visualize=False,          # Set True to see plots (slower)
    output_dir='cleaned_output'  # Cleaned files go here
)

print("\n✅ Done! All cleaned files saved to 'cleaned_output/' folder")

# View summary
print("\nResults summary:")
for filename, stats in results.items():
    if stats['status'] == 'success':
        print(f"  {filename}: Removed {stats['removed_points']:,} spikes ({stats['percent_removed']:.1f}%)")
