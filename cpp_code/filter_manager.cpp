#include "filter_manager.h"
#include <algorithm>
#include <cmath>

namespace keyence {

FilterManager::FilterManager()
    : global_z_min_value_(0.0f),
      global_z_max_value_(1.0f),
      global_z_min_bound_(0.0f),
      global_z_max_bound_(1.0f),
      next_region_id_(1) {
}

void FilterManager::setGlobalRange(float z_min, float z_max) {
    global_z_min_value_ = z_min;
    global_z_max_value_ = z_max;
}

void FilterManager::getGlobalRange(float& z_min, float& z_max) const {
    z_min = global_z_min_value_;
    z_max = global_z_max_value_;
}

void FilterManager::setDataBounds(float z_min, float z_max) {
    global_z_min_bound_ = z_min;
    global_z_max_bound_ = z_max;
}

void FilterManager::getDataBounds(float& z_min, float& z_max) const {
    z_min = global_z_min_bound_;
    z_max = global_z_max_bound_;
}

int FilterManager::addRegion(const RegionalFilter& region) {
    RegionalFilter new_region = region;
    new_region.id = next_region_id_++;
    regions_.push_back(new_region);
    return new_region.id;
}

void FilterManager::removeRegion(int id) {
    regions_.erase(
        std::remove_if(regions_.begin(), regions_.end(),
                      [id](const RegionalFilter& r) { return r.id == id; }),
        regions_.end()
    );
}

void FilterManager::updateRegion(int id, const RegionalFilter& region) {
    for (auto& r : regions_) {
        if (r.id == id) {
            r = region;
            r.id = id;  // Preserve ID
            return;
        }
    }
}

const RegionalFilter* FilterManager::getRegion(int id) const {
    for (const auto& r : regions_) {
        if (r.id == id) {
            return &r;
        }
    }
    return nullptr;
}

void FilterManager::applyFilters(const Eigen::MatrixXf& heights,
                                float pixel_pitch_x,
                                float pixel_pitch_y,
                                const std::vector<int>& row_indices,
                                const std::vector<int>& col_indices) {
    int rows = heights.rows();
    int cols = heights.cols();
    
    // Resize output matrices if needed
    if (visibility_mask_.rows() != rows || visibility_mask_.cols() != cols) {
        visibility_mask_.resize(rows, cols);
        filter_reason_.resize(rows, cols);
    }
    
    // Determine if we're using sampling
    bool using_sampling = (!row_indices.empty() && !col_indices.empty());
    
    // Process each point
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float z_mm = heights(row, col);
            
            // Skip NaN/invalid points
            if (std::isnan(z_mm)) {
                visibility_mask_(row, col) = false;
                filter_reason_(row, col) = 0;
                continue;
            }
            
            // Convert to physical coordinates
            // If sampling, map back to original coordinates
            int original_row = using_sampling ? row_indices[row] : row;
            int original_col = using_sampling ? col_indices[col] : col;
            
            float x_physical = original_col * pixel_pitch_x;  // μm
            float y_physical = original_row * pixel_pitch_y;  // μm
            float z_physical = z_mm * 1000.0f;                // mm to μm
            
            // Check if point is in any regional filter
            bool in_region = false;
            const RegionalFilter* active_region = nullptr;
            
            for (const auto& region : regions_) {
                if (region.active) {
                    if (x_physical >= region.x_min && x_physical <= region.x_max &&
                        y_physical >= region.y_min && y_physical <= region.y_max) {
                        in_region = true;
                        active_region = &region;
                        break;  // First matching region wins
                    }
                }
            }
            
            // Apply appropriate Z-range filter
            bool passes_filter;
            uint8_t reason;
            
            if (in_region) {
                // Use regional Z-range
                passes_filter = (z_physical >= active_region->z_min &&
                               z_physical <= active_region->z_max);
                reason = passes_filter ? 0 : (active_region->id + 1);
            } else {
                // Use global Z-range
                passes_filter = (z_physical >= global_z_min_value_ &&
                               z_physical <= global_z_max_value_);
                reason = passes_filter ? 0 : 1;
            }
            
            visibility_mask_(row, col) = passes_filter;
            filter_reason_(row, col) = reason;
        }
    }
}

FilterManager::Statistics FilterManager::getStatistics() const {
    Statistics stats;
    stats.total_points = visibility_mask_.size();
    stats.visible_count = 0;
    stats.global_filtered_count = 0;
    stats.regional_filtered_count = 0;
    
    for (int i = 0; i < visibility_mask_.rows(); ++i) {
        for (int j = 0; j < visibility_mask_.cols(); ++j) {
            if (visibility_mask_(i, j)) {
                stats.visible_count++;
            } else if (filter_reason_(i, j) == 1) {
                stats.global_filtered_count++;
            } else if (filter_reason_(i, j) > 1) {
                stats.regional_filtered_count++;
            }
        }
    }
    
    return stats;
}

} // namespace keyence
