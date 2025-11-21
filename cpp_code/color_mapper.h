#ifndef COLOR_MAPPER_H
#define COLOR_MAPPER_H

#include <Eigen/Dense>
#include <vtkColorTransferFunction.h>
#include <vtkSmartPointer.h>

namespace keyence {

/**
 * @brief Maps height values to colors using VTK color transfer functions
 */
class ColorMapper {
public:
    ColorMapper();
    ~ColorMapper();
    
    /**
     * @brief Compute colors for all points based on filter state
     * 
     * @param heights Height matrix (rows × cols, in mm)
     * @param visibility Visibility mask
     * @param filter_reasons Filter reason codes
     * @param show_filtered Whether to show filtered points in red/orange
     * @param z_min_range Min of color normalization range (μm)
     * @param z_max_range Max of color normalization range (μm)
     */
    void computeColors(const Eigen::MatrixXf& heights,
                      const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visibility,
                      const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& filter_reasons,
                      bool show_filtered,
                      float z_min_range,
                      float z_max_range);
    
    /**
     * @brief Get RGB color channels
     */
    const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& getRedChannel() const {
        return colors_r_;
    }
    
    const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& getGreenChannel() const {
        return colors_g_;
    }
    
    const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& getBlueChannel() const {
        return colors_b_;
    }
    
private:
    // VTK color transfer function (colormap)
    vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction_;
    
    // Color storage (same size as height matrix)
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> colors_r_;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> colors_g_;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> colors_b_;
    
    /**
     * @brief Initialize color transfer function (blue to red gradient)
     */
    void initializeColormap();
    
    /**
     * @brief Get color for a filtered point based on filter reason
     */
    void getFilteredColor(uint8_t filter_reason, 
                         unsigned char& r, 
                         unsigned char& g, 
                         unsigned char& b);
};

} // namespace keyence

#endif // COLOR_MAPPER_H
