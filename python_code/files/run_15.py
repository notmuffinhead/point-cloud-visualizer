from keyence_batch import quick_keyence_analysis_and_clean

# Process ALL CSV files in your folder!
analyzer, results = quick_keyence_analysis_and_clean(
    folder_path=r"../KEYENCE_DATASET",
    subsample=4,              # Fast! ~5 min per 51M points
    curvature_threshold=0.1,  # Removes high-curvature spikes
    n_jobs=-1,                # Use all CPU cores
    visualize=True,          # Set True to see plots
    output_dir='cleaned_output'
)

print(f"✅ Processed {len(results)} files!")