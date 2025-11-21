#include "color_mapper.h"
#include <algorithm>
#include <cmath>

namespace keyence {

ColorMapper::ColorMapper() {
    colorTransferFunction_ = vtkSmartPointer<vtkColorTransferFunction>::New();
    initializeColormap();
}

ColorMapper::~ColorMapper() {
}

void ColorMapper::initializeColormap() {
    // Create blue-to-red gradient (similar to your existing VTK setup)
    colorTransferFunction_->SetColorSpaceToRGB();
    
    // Blue to cyan to green to yellow to red gradient
    colorTransferFunction_->AddRGBPoint(0.0, 0.0, 0.0, 1.0);   // Blue
    colorTransferFunction_->AddRGBPoint(0.25, 0.0, 0.5, 1.0);  // Cyan-blue
    colorTransferFunction_->AddRGBPoint(0.5, 0.0, 1.0, 0.5);   // Green-cyan
    colorTransferFunction_->AddRGBPoint(0.75, 1.0, 1.0, 0.0);  // Yellow
    colorTransferFunction_->AddRGBPoint(1.0, 1.0, 0.0, 0.0);   // Red
}

void ColorMapper::computeColors(const Eigen::MatrixXf& heights,
                                const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visibility,
                                const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& filter_reasons,
                                bool show_filtered,
                                float z_min_range,
                                float z_max_range) {
    int rows = heights.rows();
    int cols = heights.cols();
    
    // Resize color matrices if needed
    if (colors_r_.rows() != rows || colors_r_.cols() != cols) {
        colors_r_.resize(rows, cols);
        colors_g_.resize(rows, cols);
        colors_b_.resize(rows, cols);
    }
    
    float range = z_max_range - z_min_range;
    if (range < 0.001f) {
        range = 1.0f;  // Avoid division by zero
    }
    
    // Process each point
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float z_mm = heights(row, col);
            
            // Skip NaN/invalid points
            if (std::isnan(z_mm)) {
                colors_r_(row, col) = 0;
                colors_g_(row, col) = 0;
                colors_b_(row, col) = 0;
                continue;
            }
            
            if (visibility(row, col)) {
                // Point is visible - use colormap based on height
                float z_physical = z_mm * 1000.0f;  // mm to μm
                
                // Normalize based on user-selected range
                float normalized = (z_physical - z_min_range) / range;
                
                // Clamp to [0, 1]
                normalized = std::max(0.0f, std::min(1.0f, normalized));
                
                // Get color from VTK colormap
                double rgb[3];
                colorTransferFunction_->GetColor(normalized, rgb);
                
                colors_r_(row, col) = static_cast<unsigned char>(rgb[0] * 255);
                colors_g_(row, col) = static_cast<unsigned char>(rgb[1] * 255);
                colors_b_(row, col) = static_cast<unsigned char>(rgb[2] * 255);
                
            } else if (show_filtered) {
                // Point is filtered but user wants to see it
                unsigned char r, g, b;
                getFilteredColor(filter_reasons(row, col), r, g, b);
                
                colors_r_(row, col) = r;
                colors_g_(row, col) = g;
                colors_b_(row, col) = b;
                
            } else {
                // Point is filtered and won't be rendered
                colors_r_(row, col) = 0;
                colors_g_(row, col) = 0;
                colors_b_(row, col) = 0;
            }
        }
    }
}

void ColorMapper::getFilteredColor(uint8_t filter_reason,
                                  unsigned char& r,
                                  unsigned char& g,
                                  unsigned char& b) {
    if (filter_reason == 1) {
        // Global filter - RED
        r = 255;
        g = 0;
        b = 0;
    } else {
        // Regional filter - ORANGE
        r = 255;
        g = 165;
        b = 0;
    }
}

} // namespace keyence
