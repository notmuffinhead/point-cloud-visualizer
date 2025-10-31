#ifndef KEYENCE_ANALYZER_H
#define KEYENCE_ANALYZER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <Eigen/Dense>

// Forward declarations for VTK types
class vtkStructuredGrid;
class vtkRenderWindow;

namespace keyence {

/**
 * @brief Result structure for analysis output
 */
struct AnalysisResult {
    std::string filename;
    std::string status;           // "success" or "failed"
    int rows;
    int cols;
    int valid_points;
    int invalid_points;
    Eigen::MatrixXf raw_data;      // Direct from CSV
    Eigen::MatrixXf filtered_data; // After filtering (currently same as raw)
    Eigen::MatrixXf display_data;  // Downsampled for visualization
    
    AnalysisResult() 
        : rows(0), cols(0), valid_points(0), invalid_points(0), 
          status("not_started") {}
};

/**
 * @brief Keyence profilometer CSV data analyzer
 * 
 * Analyzes surface topography data from Keyence profilometer CSV files.
 * 
 * Keyence Specifications:
 * - X pixel pitch: 2.5 μm
 * - Y pixel pitch: 2.5 μm (nominal)
 * - Z reference: center of sensor FOV
 * - X/Y origin: top left corner of image
 * - Invalid data marker: -99999.9999
 * 
 * Pipeline:
 * 1. Load CSV → Eigen::MatrixXf
 * 2. Replace invalid markers with NaN
 * 3. [FUTURE] Optional PCA-based outlier filtering
 * 4. Downsample for visualization
 * 5. Render 3D surface with VTK
 */
class KeyenceAnalyzer {
public:
    /**
     * @brief Constructor
     */
    KeyenceAnalyzer();
    
    /**
     * @brief Destructor
     */
    ~KeyenceAnalyzer();
    
    /**
     * @brief Analyze a single Keyence CSV file
     * 
     * @param filepath Path to CSV file
     * @param display_sample_size Maximum rows/cols to display (default 500)
     * @return AnalysisResult containing raw, filtered, and display data
     */
    AnalysisResult analyzeSingleFile(const std::string& filepath, 
                                     int display_sample_size = 500);
    
    /**
     * @brief Analyze all CSV files in a folder
     * 
     * @param folder_path Path to folder containing CSV files
     * @param display_sample_size Maximum rows/cols to display
     * @return Map of filename to results
     */
    std::map<std::string, AnalysisResult> analyzeAllFiles(
        const std::string& folder_path = "KEYENCE_DATASET",
        int display_sample_size = 500);
    
    // Getters
    float getInvalidValue() const { return invalid_value_; }
    float getPixelPitchX() const { return pixel_pitch_x_; }
    float getPixelPitchY() const { return pixel_pitch_y_; }
    
    // Access results
    const std::map<std::string, AnalysisResult>& getResults() const { 
        return results_; 
    }

private:
    // ═══════════════════════════════════════════════════════════════
    // STEP 1: Data Loading
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Load CSV file into Eigen matrix
     * 
     * @param filepath Path to CSV file
     * @param data Output matrix (will be resized)
     * @return true if successful, false otherwise
     */
    bool loadCSV(const std::string& filepath, Eigen::MatrixXf& data);
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 2: Invalid Point Handling
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Replace invalid values (-99999.9999) with NaN
     * 
     * @param data Matrix to process (modified in-place)
     * @return Number of invalid points replaced
     */
    int replaceInvalidValues(Eigen::MatrixXf& data);
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 3: Filtering (FUTURE: PCA Integration Point)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Apply filtering to data (currently pass-through)
     * 
     * FUTURE: This will call filterOutliersWithPCA when implemented
     * 
     * @param raw_data Input data
     * @return Filtered data (currently just a copy)
     */
    Eigen::MatrixXf applyFiltering(const Eigen::MatrixXf& raw_data);
    
    // [FUTURE METHOD - NOT YET IMPLEMENTED]
    // Eigen::MatrixXf filterOutliersWithPCA(const Eigen::MatrixXf& data,
    //                                        int tile_size = 1000,
    //                                        float mad_threshold = 10.0f);
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 4: Sampling
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Create sampled display data from full data
     * 
     * @param full_data Full dataset
     * @param display_sample_size Maximum size for display
     * @param row_indices Output: sampled row indices
     * @param col_indices Output: sampled column indices
     * @return Sampled matrix
     */
    Eigen::MatrixXf createDisplaySample(const Eigen::MatrixXf& full_data,
                                        int display_sample_size,
                                        std::vector<int>& row_indices,
                                        std::vector<int>& col_indices);
    
    // ═══════════════════════════════════════════════════════════════
    // STEP 5: Visualization
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Create 3D surface visualization using VTK
     * 
     * @param heights Surface heights in mm (sampled)
     * @param filename Filename for plot title
     * @param row_indices Original row indices used for sampling
     * @param col_indices Original column indices used for sampling
     */
    void visualizeSurface(const Eigen::MatrixXf& heights,
                         const std::string& filename,
                         const std::vector<int>& row_indices,
                         const std::vector<int>& col_indices);
    
    /**
     * @brief Create VTK structured grid from height data
     * 
     * @param heights Height matrix in mm
     * @param row_indices Row sampling indices
     * @param col_indices Column sampling indices
     * @return VTK structured grid (caller must delete)
     */
    vtkStructuredGrid* createVTKStructuredGrid(
        const Eigen::MatrixXf& heights,
        const std::vector<int>& row_indices,
        const std::vector<int>& col_indices);
    
    // ═══════════════════════════════════════════════════════════════
    // Utility Methods
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Get list of CSV files in directory
     * 
     * @param folder_path Directory path
     * @return Vector of CSV file paths (sorted)
     */
    std::vector<std::string> getCSVFiles(const std::string& folder_path);
    
    /**
     * @brief Extract filename from full path
     * 
     * @param filepath Full path
     * @return Filename only
     */
    std::string getBasename(const std::string& filepath);
    
    /**
     * @brief Print analysis statistics
     */
    void printStatistics(const AnalysisResult& result);
    
    // ═══════════════════════════════════════════════════════════════
    // Member Variables
    // ═══════════════════════════════════════════════════════════════
    
    float invalid_value_;                              // Invalid data marker
    float pixel_pitch_x_;                              // X pixel pitch in μm
    float pixel_pitch_y_;                              // Y pixel pitch in μm
    std::map<std::string, AnalysisResult> results_;    // Analysis results
};

// ═══════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Analyze a single Keyence CSV file with 3D visualization
 * 
 * @param filepath Path to CSV file
 * @param display_sample_size Max rows/cols for display (default 500)
 * @return Analysis result
 */
AnalysisResult analyzeSingleKeyenceFile(
    const std::string& filepath, 
    int display_sample_size = 500);

/**
 * @brief Analyze all Keyence CSV files in a folder
 * 
 * @param folder_path Path to folder containing CSV files
 * @param display_sample_size Max rows/cols for display
 * @return Map of filename to results
 */
std::map<std::string, AnalysisResult> analyzeAllKeyenceFiles(
    const std::string& folder_path = "KEYENCE_DATASET",
    int display_sample_size = 500);

/**
 * @brief Smart analysis - auto-detects file or folder
 * 
 * @param path Path to CSV file or folder
 * @param display_sample_size Max rows/cols for display
 */
void quickKeyenceAnalysis(const std::string& path, 
                         int display_sample_size = 500);

} // namespace keyence

#endif // KEYENCE_ANALYZER_H
