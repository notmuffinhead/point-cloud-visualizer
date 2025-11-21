#include "vtk_updater.h"
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkPointData.h>
#include <vtkCubeSource.h>
#include <vtkExtractEdges.h>
#include <vtkRenderWindow.h>
#include <cmath>
#include <limits>

namespace keyence {

VTKUpdater::VTKUpdater() 
    : render_window_(nullptr) {
    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderer_->SetBackground(1.0, 1.0, 1.0);  // White background
    
    points_ = vtkSmartPointer<vtkPoints>::New();
    polydata_ = vtkSmartPointer<vtkPolyData>::New();
    surface_actor_ = vtkSmartPointer<vtkActor>::New();
    picker_ = vtkSmartPointer<vtkCellPicker>::New();
    picker_->SetTolerance(0.005);
}

VTKUpdater::~VTKUpdater() {
}

void VTKUpdater::setRenderWindow(vtkRenderWindow* window) {
    render_window_ = window;
}

void VTKUpdater::updatePointCloud(const Eigen::MatrixXf& heights,
                                 const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visibility,
                                 const ColorMapper& colors,
                                 bool show_filtered,
                                 float pixel_pitch_x,
                                 float pixel_pitch_y,
                                 const std::vector<int>& row_indices,
                                 const std::vector<int>& col_indices) {
    // Remove old surface actor
    if (surface_actor_) {
        renderer_->RemoveActor(surface_actor_);
    }
    
    // Create new polydata
    vtkPolyData* new_polydata = createVTKPolyData(
        heights, visibility, colors, show_filtered,
        pixel_pitch_x, pixel_pitch_y, row_indices, col_indices);
    
    if (new_polydata && new_polydata->GetNumberOfPoints() > 0) {
        // Create mapper
        vtkSmartPointer<vtkPolyDataMapper> mapper = 
            vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(new_polydata);
        mapper->ScalarVisibilityOn();
        
        // Create actor
        surface_actor_ = vtkSmartPointer<vtkActor>::New();
        surface_actor_->SetMapper(mapper);
        surface_actor_->GetProperty()->SetInterpolationToPhong();
        
        // Add to renderer
        renderer_->AddActor(surface_actor_);
        
        new_polydata->Delete();
    }
}

vtkPolyData* VTKUpdater::createVTKPolyData(
    const Eigen::MatrixXf& heights,
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visibility,
    const ColorMapper& colors,
    bool show_filtered,
    float pixel_pitch_x,
    float pixel_pitch_y,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices) {
    
    int display_rows = heights.rows();
    int display_cols = heights.cols();
    
    // Determine if we're using custom indices or direct mapping
    bool using_indices = (!row_indices.empty() && !col_indices.empty());
    
    // Create VTK data structures
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnsignedCharArray> color_array = 
        vtkSmartPointer<vtkUnsignedCharArray>::New();
    color_array->SetNumberOfComponents(3);
    color_array->SetName("Colors");
    
    // Map from (display_row, display_col) to point index
    std::vector<std::vector<int>> point_map(display_rows, 
        std::vector<int>(display_cols, -1));
    
    int point_idx = 0;
    
    const auto& colors_r = colors.getRedChannel();
    const auto& colors_g = colors.getGreenChannel();
    const auto& colors_b = colors.getBlueChannel();
    
    // Add valid, visible points
    for (int i = 0; i < display_rows; ++i) {
        for (int j = 0; j < display_cols; ++j) {
            
            float z_mm = heights(i, j);
            
            // Skip invalid points
            if (std::isnan(z_mm)) {
                continue;
            }
            
            // Check if point should be rendered
            bool is_visible = visibility(i, j);
            
            if (!show_filtered && !is_visible) {
                continue;  // Skip filtered points when not showing them
            }
            
            // Calculate physical coordinates
            // If using indices, map back to original positions
            int original_row = using_indices ? row_indices[i] : i;
            int original_col = using_indices ? col_indices[j] : j;
            
            float x_um = original_col * pixel_pitch_x;
            float y_um = original_row * pixel_pitch_y;
            float z_um = z_mm * 1000.0f;
            
            // Add point
            points->InsertNextPoint(x_um, y_um, z_um);
            
            // Add color
            unsigned char color[3] = {
                colors_r(i, j),
                colors_g(i, j),
                colors_b(i, j)
            };
            color_array->InsertNextTypedTuple(color);
            
            // Track point index
            point_map[i][j] = point_idx++;
        }
    }
    
    // Create quads connecting neighboring points
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
    
    for (int i = 0; i < display_rows - 1; ++i) {
        for (int j = 0; j < display_cols - 1; ++j) {
            int p0 = point_map[i][j];
            int p1 = point_map[i][j+1];
            int p2 = point_map[i+1][j+1];
            int p3 = point_map[i+1][j];
            
            // Only create quad if all four corners exist
            if (p0 >= 0 && p1 >= 0 && p2 >= 0 && p3 >= 0) {
                vtkIdType pts[4] = {
                    static_cast<vtkIdType>(p0),
                    static_cast<vtkIdType>(p1),
                    static_cast<vtkIdType>(p2),
                    static_cast<vtkIdType>(p3)
                };
                polys->InsertNextCell(4, pts);
            }
        }
    }
    
    // Create polydata
    vtkPolyData* polydata = vtkPolyData::New();
    polydata->SetPoints(points);
    polydata->SetPolys(polys);
    polydata->GetPointData()->SetScalars(color_array);
    
    return polydata;
}

void VTKUpdater::updateRegionBoxes(const std::vector<RegionalFilter>& regions, 
                                  bool show_boxes) {
    // Remove all existing box actors
    for (auto& actor : region_box_actors_) {
        renderer_->RemoveActor(actor);
    }
    region_box_actors_.clear();
    
    if (!show_boxes) {
        return;
    }
    
    // Create new box actors
    for (const auto& region : regions) {
        if (region.active) {
            vtkActor* box_actor = createRegionBox(region);
            if (box_actor) {
                region_box_actors_.push_back(box_actor);
                renderer_->AddActor(box_actor);
            }
        }
    }
}

vtkActor* VTKUpdater::createRegionBox(const RegionalFilter& region) {
    // Calculate box parameters
    float center_x = (region.x_min + region.x_max) / 2.0f;
    float center_y = (region.y_min + region.y_max) / 2.0f;
    float center_z = (region.z_min + region.z_max) / 2.0f;
    
    float length_x = region.x_max - region.x_min;
    float length_y = region.y_max - region.y_min;
    float length_z = region.z_max - region.z_min;
    
    // Create cube
    vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
    cube->SetCenter(center_x, center_y, center_z);
    cube->SetXLength(length_x);
    cube->SetYLength(length_y);
    cube->SetZLength(length_z);
    
    // Extract edges to create wireframe
    vtkSmartPointer<vtkExtractEdges> edges = 
        vtkSmartPointer<vtkExtractEdges>::New();
    edges->SetInputConnection(cube->GetOutputPort());
    
    // Create mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = 
        vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(edges->GetOutputPort());
    
    // Create actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    // Set color based on region ID
    double r, g, b;
    getRegionBoxColor(region.id, r, g, b);
    actor->GetProperty()->SetColor(r, g, b);
    actor->GetProperty()->SetLineWidth(2.5);
    
    return actor;
}

void VTKUpdater::getRegionBoxColor(int region_id, double& r, double& g, double& b) {
    // Cycle through distinct colors
    int color_index = (region_id - 1) % 6;
    
    switch (color_index) {
        case 0: r = 0.0; g = 1.0; b = 1.0; break;  // Cyan
        case 1: r = 1.0; g = 0.0; b = 1.0; break;  // Magenta
        case 2: r = 1.0; g = 1.0; b = 0.0; break;  // Yellow
        case 3: r = 0.0; g = 1.0; b = 0.0; break;  // Green
        case 4: r = 1.0; g = 0.5; b = 0.0; break;  // Orange
        case 5: r = 0.5; g = 0.0; b = 1.0; break;  // Purple
        default: r = 1.0; g = 1.0; b = 1.0; break; // White
    }
}

bool VTKUpdater::pickPoint(int screen_x, int screen_y,
                          float& world_x, float& world_y, float& world_z) {
    if (!render_window_) {
        return false;
    }
    
    // Perform pick
    int success = picker_->Pick(screen_x, screen_y, 0, renderer_);
    
    if (success) {
        double* pos = picker_->GetPickPosition();
        world_x = static_cast<float>(pos[0]);
        world_y = static_cast<float>(pos[1]);
        world_z = static_cast<float>(pos[2]);
        return true;
    }
    
    return false;
}

} // namespace keyence
