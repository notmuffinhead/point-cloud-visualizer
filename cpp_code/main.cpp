#include "keyence_analyzer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "    Keyence Analyzer - 3D Surface Visualization (C++)     " << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "\nFeatures:" << std::endl;
    std::cout << "  ✓ Loads full dataset (all points)" << std::endl;
    std::cout << "  ✓ Samples for display (adjustable, default 500×500)" << std::endl;
    std::cout << "  ✓ Physical coordinates (2.5 μm pixel pitch)" << std::endl;
    std::cout << "  ✓ Invalid points excluded from visualization" << std::endl;
    std::cout << "  ✓ Interactive 3D visualization with VTK" << std::endl;
    std::cout << "  ⧗ PCA-based outlier filtering (coming soon)" << std::endl;
    
    std::cout << "\nPipeline:" << std::endl;
    std::cout << "  1. Load CSV → Eigen::MatrixXf" << std::endl;
    std::cout << "  2. Replace invalid markers (-99999.9999) with NaN" << std::endl;
    std::cout << "  3. Apply filtering (future: PCA outlier detection)" << std::endl;
    std::cout << "  4. Downsample for visualization" << std::endl;
    std::cout << "  5. Render 3D surface with VTK" << std::endl;
    
    if (argc < 2) {
        std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " <csv_file_or_folder> [display_sample_size]" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " data.csv" << std::endl;
        std::cout << "  " << argv[0] << " data.csv 1000" << std::endl;
        std::cout << "  " << argv[0] << " KEYENCE_DATASET" << std::endl;
        std::cout << "  " << argv[0] << " KEYENCE_DATASET 800" << std::endl;
        std::cout << "\nParameters:" << std::endl;
        std::cout << "  csv_file_or_folder:  Path to CSV file or folder containing CSVs" << std::endl;
        std::cout << "  display_sample_size: Max rows/cols to display (default: 500)" << std::endl;
        std::cout << "                       Higher = more detail, slower rendering" << std::endl;
        std::cout << "                       Recommended range: 200-1000" << std::endl;
        std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
        return 1;
    }
    
    std::string path = argv[1];
    int display_sample_size = 500;
    
    if (argc >= 3) {
        try {
            display_sample_size = std::stoi(argv[2]);
            if (display_sample_size < 50 || display_sample_size > 2000) {
                std::cerr << "Warning: display_sample_size should be between 50-2000" << std::endl;
                std::cerr << "         Using default (500)" << std::endl;
                display_sample_size = 500;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid display_sample_size, using default (500)" << std::endl;
        }
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Starting analysis..." << std::endl;
    std::cout << "  Path: " << path << std::endl;
    std::cout << "  Display sample size: " << display_sample_size << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    
    // Run the analysis
    keyence::quickKeyenceAnalysis(path, display_sample_size);
    
    return 0;
}
