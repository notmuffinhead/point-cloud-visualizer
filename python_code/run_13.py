from open3d_try1_13 import quick_keyence_analysis

# Analyze all files in custom folder
analyzer, results = quick_keyence_analysis("KEYENCE_DATASET_PYTHON_COPY")

# With custom parameters
# analyzer, results = quick_keyence_analysis("my_data", k_neighbors=30, subsample=4)

# Disable some visualizations for speed
# analyzer, results = quick_keyence_analysis(plot_3d=False, plot_histograms=False)