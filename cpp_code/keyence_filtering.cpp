// ═══════════════════════════════════════════════════════════════════
// FUTURE IMPLEMENTATION: PCL-Based Outlier Filtering
// ═══════════════════════════════════════════════════════════════════
// 
// This file contains skeleton implementations and detailed pseudocode
// for PCL-based outlier filtering using local PCA and MAD.
//
// To implement:
//   1. Install PCL: sudo apt-get install libpcl-dev
//   2. Uncomment PCL includes in keyence_filtering.h
//   3. Uncomment PCL sections in CMakeLists.txt
//   4. Implement the TODOs below
//   5. Link in keyence_analyzer.cpp
//
// ═══════════════════════════════════════════════════════════════════

#include "keyence_filtering.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// Uncomment when PCL is available:
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/common/pca.h>

namespace keyence {

// ═══════════════════════════════════════════════════════════════════
// Main Filtering Function
// ═══════════════════════════════════════════════════════════════════

Eigen::MatrixXf filterOutliersWithPCA(
    const Eigen::MatrixXf& data,
    float pixel_pitch_x,
    float pixel_pitch_y,
    int tile_size,
    float mad_threshold,
    OutlierStatistics* stats) {
    
    std::cout << "\n🔬 ADAPTIVE MAD-BASED SPIKE DETECTION" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "📊 Method: Median Absolute Deviation (scale-invariant)" << std::endl;
    std::cout << "   Tile size: " << tile_size << "×" << tile_size << " pixels" << std::endl;
    std::cout << "   MAD threshold: " << mad_threshold << "× (removes points > " 
              << mad_threshold << "×MAD from median)" << std::endl;
    
    int rows = data.rows();
    int cols = data.cols();
    
    std::cout << "   Data size: " << rows << " × " << cols << " = " 
              << (rows * cols) << " points" << std::endl;
    
    // Calculate tile grid
    int num_tiles_y = (rows + tile_size - 1) / tile_size;
    int num_tiles_x = (cols + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_y * num_tiles_x;
    
    std::cout << "   Grid: " << num_tiles_y << " × " << num_tiles_x 
              << " = " << total_tiles << " tiles" << std::endl;
    
    // Initialize result matrices
    Eigen::MatrixXf filtered_data = data;  // Start with copy
    Eigen::MatrixXf outlier_mask = Eigen::MatrixXf::Zero(rows, cols);
    
    // Statistics tracking
    OutlierStatistics local_stats;
    std::vector<float> all_mads;
    
    std::cout << "\n🔄 Processing tiles..." << std::endl;
    
    // TODO: Implement tile processing loop
    // 
    // PSEUDOCODE:
    // 
    // for tile_y = 0 to num_tiles_y:
    //     for tile_x = 0 to num_tiles_x:
    //         
    //         // 1. Extract tile bounds
    //         row_start = tile_y * tile_size
    //         row_end = min((tile_y + 1) * tile_size, rows)
    //         col_start = tile_x * tile_size
    //         col_end = min((tile_x + 1) * tile_size, cols)
    //         
    //         // 2. Extract tile data
    //         tile_data = extractTile(data, row_start, row_end, col_start, col_end)
    //         
    //         // 3. Convert tile to PCL point cloud
    //         // tile_cloud = eigenToPCLOrganized(tile_data, pixel_pitch_x, pixel_pitch_y)
    //         // Adjust X,Y coordinates by tile offset
    //         
    //         // 4. Skip if too few valid points
    //         valid_count = countValidPoints(tile_data)
    //         if valid_count < 10:
    //             continue
    //         
    //         // 5. Compute PCA for tile
    //         // pca_result = computeTilePCA(tile_cloud)
    //         // OR: pca_result = computePCAManual(points_matrix)
    //         
    //         // 6. Calculate distances from each point to best-fit plane
    //         distances = []
    //         for each valid point in tile:
    //             dist = distanceToPlane(point, pca_result)
    //             distances.append(dist)
    //         
    //         // 7. Compute MAD for this tile
    //         median_dist = computeMedian(distances)
    //         mad = computeMAD(distances)
    //         all_mads.append(mad)
    //         
    //         // Handle edge case: MAD ≈ 0 (all points coplanar)
    //         if mad < 0.01:  // < 0.01 μm
    //             mad = 0.1   // Use small default
    //         
    //         // 8. Detect outliers
    //         outliers = detectOutliersMAD(distances, mad_threshold)
    //         
    //         // 9. Mark outliers in global mask
    //         for each point index i in tile:
    //             if outliers[i]:
    //                 global_row = row_start + (i / tile_width)
    //                 global_col = col_start + (i % tile_width)
    //                 outlier_mask(global_row, global_col) = 1
    //                 filtered_data(global_row, global_col) = NaN
    //         
    //         // 10. Update statistics
    //         if any outliers in tile:
    //             local_stats.num_tiles_with_outliers++
    //         local_stats.num_tiles_processed++
    //
    // End pseudocode
    
    // Calculate final statistics
    local_stats.total_outliers = (outlier_mask.array() > 0).count();
    local_stats.total_valid_points = countValidPoints(data);
    
    if (!all_mads.empty()) {
        local_stats.mean_mad = std::accumulate(all_mads.begin(), all_mads.end(), 0.0f) 
                               / all_mads.size();
        local_stats.median_mad = computeMedian(all_mads);
    }
    
    // Print results
    printFilteringStatistics(local_stats);
    
    if (stats) {
        *stats = local_stats;
    }
    
    return filtered_data;
}

// ═══════════════════════════════════════════════════════════════════
// Conversion Functions
// ═══════════════════════════════════════════════════════════════════

// TODO: Implement when PCL is available
/*
pcl::PointCloud<pcl::PointXYZ>::Ptr eigenToPCLOrganized(
    const Eigen::MatrixXf& data,
    float pixel_pitch_x,
    float pixel_pitch_y) {
    
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    
    // Set organized structure
    cloud->width = data.cols();
    cloud->height = data.rows();
    cloud->is_dense = false;  // Has NaN points
    cloud->points.resize(cloud->width * cloud->height);
    
    // Fill points
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            pcl::PointXYZ& pt = cloud->at(j, i);  // PCL uses (col, row)
            
            // Physical coordinates
            pt.x = j * pixel_pitch_x;      // μm
            pt.y = i * pixel_pitch_y;      // μm
            pt.z = data(i, j) * 1000.0f;   // mm → μm
            
            // PCL handles NaN automatically
        }
    }
    
    return cloud;
}
*/

Eigen::MatrixXf extractTile(
    const Eigen::MatrixXf& data,
    int row_start, int row_end,
    int col_start, int col_end) {
    
    int tile_rows = row_end - row_start;
    int tile_cols = col_end - col_start;
    
    return data.block(row_start, col_start, tile_rows, tile_cols);
}

// ═══════════════════════════════════════════════════════════════════
// PCA Analysis Functions
// ═══════════════════════════════════════════════════════════════════

// TODO: Implement when PCL is available
/*
PCAResult computeTilePCA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    PCAResult result;
    
    // Use PCL's PCA class
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    
    // Get results
    result.eigenvalues = pca.getEigenValues();
    result.eigenvectors = pca.getEigenVectors();
    
    Eigen::Vector4f mean = pca.getMean();
    result.centroid = mean.head<3>();
    
    // Normal is eigenvector with smallest eigenvalue (last column)
    result.normal = result.eigenvectors.col(2);
    
    // Plane equation: normal · (p - centroid) = 0
    // Rearrange: normal · p = normal · centroid
    result.d = -result.normal.dot(result.centroid);
    
    return result;
}
*/

PCAResult computePCAManual(const Eigen::MatrixX3f& points) {
    PCAResult result;
    
    if (points.rows() < 3) {
        return result;  // Not enough points
    }
    
    // 1. Compute centroid
    result.centroid = points.colwise().mean();
    
    // 2. Center the points
    Eigen::MatrixX3f centered = points.rowwise() - result.centroid.transpose();
    
    // 3. Compute covariance matrix
    Eigen::Matrix3f covariance = (centered.transpose() * centered) / (points.rows() - 1);
    
    // 4. Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Warning: Eigen decomposition failed" << std::endl;
        return result;
    }
    
    // Eigenvalues in ascending order, we want descending
    result.eigenvalues = solver.eigenvalues().reverse();
    
    // Eigenvectors corresponding to reversed eigenvalues
    result.eigenvectors.col(0) = solver.eigenvectors().col(2);  // Largest
    result.eigenvectors.col(1) = solver.eigenvectors().col(1);  // Middle
    result.eigenvectors.col(2) = solver.eigenvectors().col(0);  // Smallest
    
    // Normal is eigenvector with smallest eigenvalue
    result.normal = result.eigenvectors.col(2);
    
    // Ensure normal points "up" (positive Z component)
    if (result.normal.z() < 0) {
        result.normal = -result.normal;
    }
    
    // Plane equation
    result.d = -result.normal.dot(result.centroid);
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// Outlier Detection Functions
// ═══════════════════════════════════════════════════════════════════

float distanceToPlane(const Eigen::Vector3f& point, const PCAResult& pca) {
    // Distance = |normal · point + d| / ||normal||
    // If normal is unit vector, ||normal|| = 1
    
    float distance = pca.normal.dot(point) + pca.d;
    return distance / pca.normal.norm();
}

float computeMedian(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n/2, values.end());
    
    if (n % 2 == 0) {
        float mid1 = values[n/2];
        std::nth_element(values.begin(), values.begin() + n/2 - 1, values.end());
        float mid2 = values[n/2 - 1];
        return (mid1 + mid2) / 2.0f;
    } else {
        return values[n/2];
    }
}

float computeMAD(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    // 1. Compute median
    float median = computeMedian(values);
    
    // 2. Compute absolute deviations
    std::vector<float> abs_deviations;
    abs_deviations.reserve(values.size());
    
    for (float val : values) {
        abs_deviations.push_back(std::abs(val - median));
    }
    
    // 3. Return median of absolute deviations
    return computeMedian(abs_deviations);
}

std::vector<bool> detectOutliersMAD(
    const std::vector<float>& distances,
    float mad_threshold) {
    
    std::vector<bool> outliers(distances.size(), false);
    
    if (distances.empty()) {
        return outliers;
    }
    
    // Compute median and MAD
    float median = computeMedian(distances);
    float mad = computeMAD(distances);
    
    // Handle edge case
    if (mad < 0.01f) {
        mad = 0.1f;  // Small default
    }
    
    // Mark outliers
    for (size_t i = 0; i < distances.size(); ++i) {
        float deviation_ratio = std::abs(distances[i] - median) / mad;
        
        if (deviation_ratio > mad_threshold) {
            outliers[i] = true;
        }
    }
    
    return outliers;
}

// ═══════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════

int countValidPoints(const Eigen::MatrixXf& data) {
    int count = 0;
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            if (!std::isnan(data(i, j))) {
                count++;
            }
        }
    }
    return count;
}

void printFilteringStatistics(const OutlierStatistics& stats) {
    std::cout << "\n📊 Tile statistics:" << std::endl;
    std::cout << "   Tiles processed: " << stats.num_tiles_processed << std::endl;
    std::cout << "   Tiles with spikes: " << stats.num_tiles_with_outliers << std::endl;
    std::cout << "   Mean MAD: " << stats.mean_mad << " μm" << std::endl;
    std::cout << "   Median MAD: " << stats.median_mad << " μm" << std::endl;
    
    std::cout << "\n🎯 Spike detection results:" << std::endl;
    std::cout << "   Spikes detected: " << stats.total_outliers << " / " 
              << stats.total_valid_points;
    
    if (stats.total_valid_points > 0) {
        float percentage = 100.0f * stats.total_outliers / stats.total_valid_points;
        std::cout << " (" << percentage << "%)" << std::endl;
    } else {
        std::cout << std::endl;
    }
}

} // namespace keyence
