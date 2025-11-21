#ifndef FILTER_MANAGER_H
#define FILTER_MANAGER_H

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace keyence {

/**
 * @brief Regional filter definition
 */
struct RegionalFilter {
    bool active;
    std::string name;
    
    // Physical coordinates (μm)
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
    
    int id;
    
    RegionalFilter() 
        : active(true), name("Region"), 
          x_min(0), x_max(0), y_min(0), y_max(0), z_min(0), z_max(0),
          id(0) {}
};

/**
 * @brief Manages filter state and computes point visibility
 */
class FilterManager {
public:
    FilterManager();
    
    /**
     * @brief Set global Z-range filter
     */
    void setGlobalRange(float z_min, float z_max);
    
    /**
     * @brief Get current global range
     */
    void getGlobalRange(float& z_min, float& z_max) const;
    
    /**
     * @brief Set global data bounds (computed from dataset)
     */
    void setDataBounds(float z_min, float z_max);
    
    /**
     * @brief Get global data bounds
     */
    void getDataBounds(float& z_min, float& z_max) const;
    
    /**
     * @brief Add a regional filter
     * @return ID of added region
     */
    int addRegion(const RegionalFilter& region);
    
    /**
     * @brief Remove region by ID
     */
    void removeRegion(int id);
    
    /**
     * @brief Update existing region
     */
    void updateRegion(int id, const RegionalFilter& region);
    
    /**
     * @brief Get region by ID
     */
    const RegionalFilter* getRegion(int id) const;
    
    /**
     * @brief Get all regions
     */
    const std::vector<RegionalFilter>& getRegions() const { return regions_; }
    
    /**
     * @brief Apply filters to height data
     * 
     * @param heights Height matrix (rows × cols, in mm)
     * @param pixel_pitch_x X pixel pitch in μm
     * @param pixel_pitch_y Y pixel pitch in μm
     * @param row_indices Original row indices (for mapping back to full coords)
     * @param col_indices Original column indices (for mapping back to full coords)
     */
    void applyFilters(const Eigen::MatrixXf& heights,
                     float pixel_pitch_x,
                     float pixel_pitch_y,
                     const std::vector<int>& row_indices,
                     const std::vector<int>& col_indices);
    
    /**
     * @brief Get visibility mask (true = visible, false = filtered)
     */
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& getVisibilityMask() const {
        return visibility_mask_;
    }
    
    /**
     * @brief Get filter reasons
     * 0 = visible
     * 1 = filtered by global range
     * 2+ = filtered by region (region_id + 1)
     */
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& getFilterReasons() const {
        return filter_reason_;
    }
    
    /**
     * @brief Get statistics about filtering
     */
    struct Statistics {
        int total_points;
        int visible_count;
        int global_filtered_count;
        int regional_filtered_count;
    };
    
    Statistics getStatistics() const;
    
private:
    // Global Z-range (user-selected)
    float global_z_min_value_;
    float global_z_max_value_;
    
    // Data bounds (computed from dataset)
    float global_z_min_bound_;
    float global_z_max_bound_;
    
    // Regional filters
    std::vector<RegionalFilter> regions_;
    int next_region_id_;
    
    // Filter results (same size as height matrix)
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> visibility_mask_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> filter_reason_;
};

} // namespace keyence

#endif // FILTER_MANAGER_H
