#ifndef VTK_UPDATER_H
#define VTK_UPDATER_H

#include <Eigen/Dense>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCellArray.h>
#include <vtkActor.h>
#include <vtkCellPicker.h>
#include <vector>
#include "filter_manager.h"
#include "color_mapper.h"

namespace keyence {

/**
 * @brief Manages VTK visualization with filtered point cloud data
 */
class VTKUpdater {
public:
    VTKUpdater();
    ~VTKUpdater();
    
    /**
     * @brief Update point cloud visualization with filtered data
     * 
     * @param heights Height matrix (mm)
     * @param visibility Visibility mask
     * @param colors Color mapper with computed colors
     * @param show_filtered Whether filtered points should be rendered
     * @param pixel_pitch_x X pixel pitch (μm)
     * @param pixel_pitch_y Y pixel pitch (μm)
     * @param row_indices Indices for sampling (empty = use all)
     * @param col_indices Indices for sampling (empty = use all)
     */
    void updatePointCloud(const Eigen::MatrixXf& heights,
                         const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visibility,
                         const ColorMapper& colors,
                         bool show_filtered,
                         float pixel_pitch_x,
                         float pixel_pitch_y,
                         const std::vector<int>& row_indices = std::vector<int>(),
                         const std::vector<int>& col_indices = std::vector<int>());
    
    /**
     * @brief Update wireframe boxes for regional filters
     * 
     * @param regions List of regional filters
     * @param show_boxes Whether to show boxes
     */
    void updateRegionBoxes(const std::vector<RegionalFilter>& regions, bool show_boxes);
    
    /**
     * @brief Pick a point in 3D from screen coordinates
     * 
     * @param screen_x Screen X coordinate
     * @param screen_y Screen Y coordinate  
     * @param world_x Output: World X coordinate (μm)
     * @param world_y Output: World Y coordinate (μm)
     * @param world_z Output: World Z coordinate (μm)
     * @return true if pick was successful
     */
    bool pickPoint(int screen_x, int screen_y,
                  float& world_x, float& world_y, float& world_z);
    
    /**
     * @brief Get the VTK renderer
     */
    vtkRenderer* getRenderer() { return renderer_; }
    
    /**
     * @brief Set the render window (needed for picking)
     */
    void setRenderWindow(vtkRenderWindow* window);
    
private:
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkPoints> points_;
    vtkSmartPointer<vtkPolyData> polydata_;
    vtkSmartPointer<vtkActor> surface_actor_;
    vtkSmartPointer<vtkCellPicker> picker_;
    vtkRenderWindow* render_window_;
    
    std::vector<vtkSmartPointer<vtkActor>> region_box_actors_;
    
    /**
     * @brief Create VTK polydata from filtered point cloud
     */
    vtkPolyData* createVTKPolyData(
        const Eigen::MatrixXf& heights,
        const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visibility,
        const ColorMapper& colors,
        bool show_filtered,
        float pixel_pitch_x,
        float pixel_pitch_y,
        const std::vector<int>& row_indices,
        const std::vector<int>& col_indices);
    
    /**
     * @brief Create wireframe box actor for a region
     */
    vtkActor* createRegionBox(const RegionalFilter& region);
    
    /**
     * @brief Get color for region box based on ID
     */
    void getRegionBoxColor(int region_id, double& r, double& g, double& b);
};

} // namespace keyence

#endif // VTK_UPDATER_H
