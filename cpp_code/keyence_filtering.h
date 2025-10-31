#ifndef KEYENCE_FILTERING_H
#define KEYENCE_FILTERING_H

// ═══════════════════════════════════════════════════════════════════
// FUTURE IMPLEMENTATION: PCL-Based Outlier Filtering
// ═══════════════════════════════════════════════════════════════════
// 
// This file contains the skeleton for future PCL integration.
// To enable:
//   1. Uncomment PCL sections in CMakeLists.txt
//   2. Implement the functions in keyence_filtering.cpp
//   3. Include this header in keyence_analyzer.cpp
//   4. Call filterOutliersWithPCA from applyFiltering()
//
// Dependencies: PCL (Point Cloud Library) 1.8+
//   sudo apt-get install libpcl-dev
//
// ═══════════════════════════════════════════════════════════════════

#include <Eigen/Dense>

// Uncomment when PCL is available:
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/common/pca.h>
// #include <pcl/search/kdtree.h>

namespace keyence {

/**
 * @brief PCA result for a tile or local region
 */
struct PCAResult {
    Eigen::Vector3f centroid;      // Center of the tile
    Eigen::Vector3f normal;        // Surface normal (smallest eigenvector)
    Eigen::Vector3f eigenvalues;   // [λ₀, λ₁, λ₂] (sorted large to small)
    Eigen::Matrix3f eigenvectors;  // Corresponding eigenvectors
    
    // Plane representation: normal · (p - centroid) = 0
    float d;  // Plane constant: d = -normal · centroid
    
    PCAResult() : d(0.0f) {
        centroid.setZero();
        normal.setZero();
        eigenvalues.setZero();
        eigenvectors.setIdentity();
    }
};

/**
 * @brief Statistics for outlier detection
 */
struct OutlierStatistics {
    int num_tiles_processed;
    int num_tiles_with_outliers;
    int total_outliers;
    int total_valid_points;
    float mean_mad;           // Mean MAD across all tiles
    float median_mad;         // Median MAD across all tiles
    
    OutlierStatistics() 
        : num_tiles_processed(0), num_tiles_with_outliers(0),
          total_outliers(0), total_valid_points(0),
          mean_mad(0.0f), median_mad(0.0f) {}
};

// ═══════════════════════════════════════════════════════════════════
// Main Filtering Function
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Filter outliers using tile-based PCA with MAD
 * 
 * Algorithm:
 * 1. Divide surface into tiles
 * 2. For each tile:
 *    a. Convert to PCL organized point cloud
 *    b. Compute local PCA (best-fit plane)
 *    c. Calculate distance of each point to plane
 *    d. Compute MAD (Median Absolute Deviation)
 *    e. Mark points with |distance - median| / MAD > threshold
 * 3. Return matrix with outliers set to NaN
 * 
 * @param data Input height matrix (mm)
 * @param pixel_pitch_x X spacing in μm
 * @param pixel_pitch_y Y spacing in μm
 * @param tile_size Tile size in pixels (default 1000)
 * @param mad_threshold MAD multiplier (default 10.0)
 * @param stats Output statistics (optional)
 * @return Filtered matrix with outliers as NaN
 */
Eigen::MatrixXf filterOutliersWithPCA(
    const Eigen::MatrixXf& data,
    float pixel_pitch_x,
    float pixel_pitch_y,
    int tile_size = 1000,
    float mad_threshold = 10.0f,
    OutlierStatistics* stats = nullptr);

// ═══════════════════════════════════════════════════════════════════
// Conversion Functions (Eigen ↔ PCL)
// ═══════════════════════════════════════════════════════════════════

// Uncomment when PCL is available:
/*
 * @brief Convert Eigen matrix to PCL organized point cloud
 * 
 * Preserves the row/column structure:
 * - cloud->width = data.cols()
 * - cloud->height = data.rows()
 * - cloud->points[i*width + j] corresponds to data(i,j)
 * 
 * @param data Height matrix (mm)
 * @param pixel_pitch_x X spacing (μm)
 * @param pixel_pitch_y Y spacing (μm)
 * @return Organized PCL point cloud
 *
pcl::PointCloud<pcl::PointXYZ>::Ptr eigenToPCLOrganized(
    const Eigen::MatrixXf& data,
    float pixel_pitch_x,
    float pixel_pitch_y);
*/

/**
 * @brief Extract a tile from the full matrix
 * 
 * @param data Full height matrix
 * @param row_start Starting row index
 * @param row_end Ending row index (exclusive)
 * @param col_start Starting column index
 * @param col_end Ending column index (exclusive)
 * @return Tile as a matrix
 */
Eigen::MatrixXf extractTile(
    const Eigen::MatrixXf& data,
    int row_start, int row_end,
    int col_start, int col_end);

// ═══════════════════════════════════════════════════════════════════
// PCA Analysis Functions
// ═══════════════════════════════════════════════════════════════════

// Uncomment when PCL is available:
/*
 * @brief Compute PCA for a point cloud tile
 * 
 * Uses PCL's PCA class to compute:
 * - Centroid (mean position)
 * - Eigenvalues (variance along principal directions)
 * - Eigenvectors (principal directions)
 * - Surface normal (eigenvector with smallest eigenvalue)
 * 
 * @param cloud Input point cloud (organized or unorganized)
 * @return PCA result structure
 *
PCAResult computeTilePCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
*/

/**
 * @brief Compute PCA manually from point matrix
 * 
 * Alternative to PCL-based PCA. Computes covariance matrix
 * and performs eigen decomposition using Eigen.
 * 
 * @param points Nx3 matrix of point coordinates
 * @return PCA result structure
 */
PCAResult computePCAManual(const Eigen::MatrixX3f& points);

// ═══════════════════════════════════════════════════════════════════
// Outlier Detection Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Calculate distance from point to plane
 * 
 * Distance = |normal · point + d| / ||normal||
 * 
 * @param point 3D point coordinates
 * @param pca PCA result containing plane equation
 * @return Signed distance (positive = above plane)
 */
float distanceToPlane(const Eigen::Vector3f& point, const PCAResult& pca);

/**
 * @brief Compute Median Absolute Deviation (MAD)
 * 
 * MAD = median(|distances - median(distances)|)
 * 
 * Scale-invariant robust measure of spread.
 * 
 * @param values Vector of values
 * @return MAD value
 */
float computeMAD(const std::vector<float>& values);

/**
 * @brief Compute median of a vector
 * 
 * @param values Input values (will be sorted)
 * @return Median value
 */
float computeMedian(std::vector<float> values);

/**
 * @brief Detect outliers in a tile using MAD
 * 
 * @param distances Distance of each point to plane
 * @param mad_threshold Threshold multiplier
 * @return Boolean mask (true = outlier)
 */
std::vector<bool> detectOutliersMAD(
    const std::vector<float>& distances,
    float mad_threshold);

// ═══════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Count valid (non-NaN) points in matrix
 */
int countValidPoints(const Eigen::MatrixXf& data);

/**
 * @brief Print filtering statistics
 */
void printFilteringStatistics(const OutlierStatistics& stats);

} // namespace keyence

#endif // KEYENCE_FILTERING_H
