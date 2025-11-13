#include "keyence_analyzer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

// VTK includes for visualization
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkStructuredGrid.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkLookupTable.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkGeometryFilter.h>
#include <vtkQuad.h>
#include <vtkCellArray.h>

namespace fs = std::filesystem;

namespace keyence {

// ═══════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ═══════════════════════════════════════════════════════════════════

KeyenceAnalyzer::KeyenceAnalyzer() 
    : invalid_value_(-99999.9999f),
      pixel_pitch_x_(2.5f),
      pixel_pitch_y_(2.5f) {
}

KeyenceAnalyzer::~KeyenceAnalyzer() {
}

// ═══════════════════════════════════════════════════════════════════
// STEP 1: Data Loading
// ═══════════════════════════════════════════════════════════════════

bool KeyenceAnalyzer::loadCSV(const std::string& filepath, Eigen::MatrixXf& data) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return false;
    }
    
    // First pass: count rows and columns
    std::vector<std::vector<float>> temp_data;
    std::string line;
    int num_cols = 0;
    
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid number in CSV: " << cell << std::endl;
                row.push_back(std::numeric_limits<float>::quiet_NaN());
            }
        }
        
        if (!row.empty()) {
            if (num_cols == 0) {
                num_cols = row.size();
            } else if (row.size() != num_cols) {
                std::cerr << "Warning: Inconsistent row length at row " 
                         << temp_data.size() << std::endl;
            }
            temp_data.push_back(row);
        }
    }
    
    file.close();
    
    if (temp_data.empty()) {
        std::cerr << "Error: No data loaded from file" << std::endl;
        return false;
    }
    
    // Convert to Eigen matrix
    int num_rows = temp_data.size();
    data.resize(num_rows, num_cols);
    
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < std::min(num_cols, (int)temp_data[i].size()); ++j) {
            data(i, j) = temp_data[i][j];
        }
    }
    
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 2: Invalid Point Handling
// ═══════════════════════════════════════════════════════════════════

int KeyenceAnalyzer::replaceInvalidValues(Eigen::MatrixXf& data) {
    int count = 0;
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            if (std::abs(data(i, j) - invalid_value_) < 0.001f) {
                data(i, j) = nan_value;
                count++;
            }
        }
    }
    
    return count;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 3: Filtering (FUTURE: PCA Integration Point)
// ═══════════════════════════════════════════════════════════════════

Eigen::MatrixXf KeyenceAnalyzer::applyFiltering(const Eigen::MatrixXf& raw_data) {
    // FUTURE: This is where PCA-based outlier detection will be integrated
    return raw_data;  // Pass-through for now
}

// ═══════════════════════════════════════════════════════════════════
// STEP 4: Sampling
// ═══════════════════════════════════════════════════════════════════

Eigen::MatrixXf KeyenceAnalyzer::createDisplaySample(
    const Eigen::MatrixXf& full_data,
    int display_sample_size,
    std::vector<int>& row_indices,
    std::vector<int>& col_indices) {
    
    int max_rows = std::min(display_sample_size, (int)full_data.rows());
    int max_cols = std::min(display_sample_size, (int)full_data.cols());
    
    // Create linearly spaced indices
    row_indices.clear();
    col_indices.clear();
    
    if (max_rows == 1) {
        row_indices.push_back(0);
    } else {
        for (int i = 0; i < max_rows; ++i) {
            int idx = (i * (full_data.rows() - 1)) / (max_rows - 1);
            row_indices.push_back(idx);
        }
    }
    
    if (max_cols == 1) {
        col_indices.push_back(0);
    } else {
        for (int j = 0; j < max_cols; ++j) {
            int idx = (j * (full_data.cols() - 1)) / (max_cols - 1);
            col_indices.push_back(idx);
        }
    }
    
    // Extract sampled data
    Eigen::MatrixXf display_data(max_rows, max_cols);
    
    for (int i = 0; i < max_rows; ++i) {
        for (int j = 0; j < max_cols; ++j) {
            display_data(i, j) = full_data(row_indices[i], col_indices[j]);
        }
    }
    
    return display_data;
}

// ═══════════════════════════════════════════════════════════════════
// STEP 5: Visualization
// ═══════════════════════════════════════════════════════════════════

vtkPolyData* KeyenceAnalyzer::createVTKPolyData(
    const Eigen::MatrixXf& heights,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices) {
    
    int rows = heights.rows();
    int cols = heights.cols();
    
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkFloatArray> scalars = vtkSmartPointer<vtkFloatArray>::New();
    scalars->SetName("Height");
    
    // Map from (i,j) to point index
    std::vector<std::vector<int>> point_map(rows, std::vector<int>(cols, -1));
    int point_idx = 0;
    
    // Add valid points
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float z_mm = heights(i, j);
            if (!std::isnan(z_mm)) {
                float x_um = col_indices[j] * pixel_pitch_x_;
                float y_um = row_indices[i] * pixel_pitch_y_;
                float z_um = z_mm * 1000.0f;
                
                points->InsertNextPoint(x_um, y_um, z_um);
                scalars->InsertNextValue(z_um);
                point_map[i][j] = point_idx++;
            }
        }
    }
    
    // Create quads for valid cells
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
    
    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols - 1; ++j) {
            int p0 = point_map[i][j];
            int p1 = point_map[i][j+1];
            int p2 = point_map[i+1][j+1];
            int p3 = point_map[i+1][j];
            
            if (p0 >= 0 && p1 >= 0 && p2 >= 0 && p3 >= 0) {
                vtkIdType pts[4] = {p0, p1, p2, p3};
                polys->InsertNextCell(4, pts);
            }
        }
    }
    
    vtkPolyData* polydata = vtkPolyData::New();
    polydata->SetPoints(points);
    polydata->SetPolys(polys);
    polydata->GetPointData()->SetScalars(scalars);
    
    return polydata;
}

void KeyenceAnalyzer::visualizeSurface(
    const Eigen::MatrixXf& heights,
    const std::string& filename,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices) {
    
    std::cout << "\nCreating 3D surface visualization..." << std::endl;
    
    // Create polydata (with gaps for NaN)
    vtkPolyData* polydata = createVTKPolyData(heights, row_indices, col_indices);

    // Create mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = 
        vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polydata);

    
    // Create color lookup table (Viridis-like)
    vtkSmartPointer<vtkLookupTable> lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetNumberOfTableValues(256);
    lut->SetHueRange(0.6, 0.0);  // Blue to red
    lut->SetSaturationRange(1.0, 1.0);
    lut->SetValueRange(0.3, 1.0);
    lut->Build();
    
    mapper->SetLookupTable(lut);
    
    double range[2];
    polydata->GetScalarRange(range);
    mapper->SetScalarRange(range);
    
    // Create actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetInterpolationToPhong();
    
    // Create scalar bar
    vtkSmartPointer<vtkScalarBarActor> scalarBar = 
        vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetLookupTable(lut);
    scalarBar->SetTitle("Height (μm)");
    scalarBar->SetNumberOfLabels(5);
    scalarBar->GetTitleTextProperty()->SetColor(0, 0, 0);
    scalarBar->GetLabelTextProperty()->SetColor(0, 0, 0);
    
    // Create renderer
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->AddViewProp(scalarBar);
    renderer->SetBackground(1.0, 1.0, 1.0);  // White background
    
    // Set camera position
    vtkSmartPointer<vtkCamera> camera = renderer->GetActiveCamera();
    camera->SetPosition(1, -1, 1);
    camera->SetFocalPoint(0, 0, 0);
    camera->SetViewUp(0, 0, 1);
    renderer->ResetCamera();
    
    // Create render window
    vtkSmartPointer<vtkRenderWindow> renderWindow = 
        vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1000, 800);
    renderWindow->SetWindowName(("3D Surface Topography: " + filename).c_str());
    
    // Create axes widget
    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
    vtkSmartPointer<vtkOrientationMarkerWidget> widget = 
        vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    widget->SetOutlineColor(0.93, 0.57, 0.13);
    widget->SetOrientationMarker(axes);
    
    // Create interactor
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = 
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);
    
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = 
        vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    interactor->SetInteractorStyle(style);
    
    widget->SetInteractor(interactor);
    widget->SetViewport(0.0, 0.0, 0.2, 0.2);
    widget->SetEnabled(1);
    widget->InteractiveOn();
    
    // Start interaction
    renderWindow->Render();
    std::cout << "3D surface plot created successfully!" << std::endl;
    std::cout << "    (Close the window to continue)" << std::endl;
    interactor->Start();

    // Clean up polydata
    polydata->Delete();
}

// ═══════════════════════════════════════════════════════════════════
// Main Analysis Function - The Pipeline
// ═══════════════════════════════════════════════════════════════════

AnalysisResult KeyenceAnalyzer::analyzeSingleFile(
    const std::string& filepath, 
    int display_sample_size) {
    
    AnalysisResult result;
    result.filename = getBasename(filepath);
    result.status = "failed";
    
    std::cout << "KEYENCE 3D SURFACE ANALYSIS" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "File: " << result.filename << std::endl;
    
    try {
        // ═══════════════════════════════════════════════════════════
        // STEP 1: Load CSV
        // ═══════════════════════════════════════════════════════════
        Eigen::MatrixXf raw_data;
        if (!loadCSV(filepath, raw_data)) {
            std::cerr << "Error loading file" << std::endl;
            return result;
        }
        
        result.rows = raw_data.rows();
        result.cols = raw_data.cols();
        
        std::cout << "Full dataset: " << result.rows << " rows × " 
                  << result.cols << " columns" << std::endl;
        std::cout << "   Total points: " << (result.rows * result.cols) << std::endl;
        
        // ═══════════════════════════════════════════════════════════
        // STEP 2: Handle invalid points
        // ═══════════════════════════════════════════════════════════
        int num_invalid = replaceInvalidValues(raw_data);
        result.invalid_points = num_invalid;
        result.valid_points = (result.rows * result.cols) - num_invalid;
        
        int total_points = result.rows * result.cols;
        std::cout << "Invalid points: " << num_invalid << "/" << total_points 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (100.0 * num_invalid / total_points) << "%)" << std::endl;
        
        result.raw_data = raw_data;
        
        // ═══════════════════════════════════════════════════════════
        // STEP 3: Apply filtering (FUTURE: PCA integration point)
        // ═══════════════════════════════════════════════════════════
        std::cout << "Applying filtering..." << std::endl;
        Eigen::MatrixXf filtered_data = applyFiltering(raw_data);
        std::cout << "   (Currently: pass-through, no filtering applied)" << std::endl;
        
        result.filtered_data = filtered_data;
        
        // ═══════════════════════════════════════════════════════════
        // STEP 4: Create display sample
        // ═══════════════════════════════════════════════════════════
        std::vector<int> row_indices, col_indices;
        Eigen::MatrixXf display_data = createDisplaySample(
            filtered_data, display_sample_size, row_indices, col_indices);
        
        std::cout << "Display sample: " << display_data.rows() << " x " 
                  << display_data.cols() << " = " 
                  << (display_data.rows() * display_data.cols()) << " points" << std::endl;
        std::cout << "   (Sampled from full dataset for performance)" << std::endl;
        
        result.display_data = display_data;
        
        // Calculate physical dimensions
        float physical_x = result.cols * pixel_pitch_x_;  // μm
        float physical_y = result.rows * pixel_pitch_y_;  // μm
        
        std::cout << "Physical dimensions: " << std::fixed << std::setprecision(1)
                  << physical_x << " x " << physical_y << " μm" << std::endl;
        
        result.status = "success";
        
        // ═══════════════════════════════════════════════════════════
        // STEP 5: Visualize surface
        // ═══════════════════════════════════════════════════════════
        visualizeSurface(display_data, result.filename, row_indices, col_indices);
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error analyzing " << result.filename << ": " 
                  << e.what() << std::endl;
        return result;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Batch Processing
// ═══════════════════════════════════════════════════════════════════

std::map<std::string, AnalysisResult> KeyenceAnalyzer::analyzeAllFiles(
    const std::string& folder_path, 
    int display_sample_size) {
    
    std::vector<std::string> csv_files = getCSVFiles(folder_path);
    
    if (csv_files.empty()) {
        std::cerr << "No CSV files found in " << folder_path << std::endl;
        return results_;
    }
    
    std::cout << "3D ANALYSIS OF ALL FILES" << std::endl;
    std::cout << "Found " << csv_files.size() << " CSV files" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    for (size_t i = 0; i < csv_files.size(); ++i) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "ANALYZING FILE " << (i + 1) << " OF " << csv_files.size() << std::endl;
        std::cout << "============================================================" << std::endl;
        
        AnalysisResult result = analyzeSingleFile(csv_files[i], display_sample_size);
        results_[result.filename] = result;
        
        if (i < csv_files.size() - 1) {
            std::cout << "\nPress Enter to continue to next file (" 
                     << (i + 2) << "/" << csv_files.size() << ")..." << std::endl;
            std::cin.ignore();
        }
    }
    
    // Summary
    int success_count = 0;
    int failed_count = 0;
    
    for (const auto& pair : results_) {
        if (pair.second.status == "success") {
            success_count++;
        } else {
            failed_count++;
        }
    }
    
    std::cout << "\nANALYSIS COMPLETE!" << std::endl;
    std::cout << "Successfully analyzed: " << success_count << " files" << std::endl;
    std::cout << "Failed analyses: " << failed_count << " files" << std::endl;
    
    return results_;
}

// ═══════════════════════════════════════════════════════════════════
// Utility Methods
// ═══════════════════════════════════════════════════════════════════

std::vector<std::string> KeyenceAnalyzer::getCSVFiles(const std::string& folder_path) {
    std::vector<std::string> csv_files;
    
    try {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file() && 
                entry.path().extension() == ".csv") {
                csv_files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    
    std::sort(csv_files.begin(), csv_files.end());
    return csv_files;
}

std::string KeyenceAnalyzer::getBasename(const std::string& filepath) {
    fs::path p(filepath);
    return p.filename().string();
}

void KeyenceAnalyzer::printStatistics(const AnalysisResult& result) {
    std::cout << "\nAnalysis Statistics:" << std::endl;
    std::cout << "  File: " << result.filename << std::endl;
    std::cout << "  Status: " << result.status << std::endl;
    std::cout << "  Dimensions: " << result.rows << " × " << result.cols << std::endl;
    std::cout << "  Valid points: " << result.valid_points << std::endl;
    std::cout << "  Invalid points: " << result.invalid_points << std::endl;
}

// ═══════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════

AnalysisResult analyzeSingleKeyenceFile(
    const std::string& filepath, 
    int display_sample_size) {
    
    KeyenceAnalyzer analyzer;
    return analyzer.analyzeSingleFile(filepath, display_sample_size);
}

std::map<std::string, AnalysisResult> analyzeAllKeyenceFiles(
    const std::string& folder_path,
    int display_sample_size) {
    
    KeyenceAnalyzer analyzer;
    return analyzer.analyzeAllFiles(folder_path, display_sample_size);
}

void quickKeyenceAnalysis(const std::string& path, int display_sample_size) {
    fs::path p(path);
    
    if (fs::is_regular_file(p) && p.extension() == ".csv") {
        std::cout << "Single file detected" << std::endl;
        analyzeSingleKeyenceFile(path, display_sample_size);
    } else if (fs::is_directory(p)) {
        std::cout << "Folder detected - analyzing all CSV files" << std::endl;
        analyzeAllKeyenceFiles(path, display_sample_size);
    } else {
        std::cerr << "Invalid path: " << path << std::endl;
    }
}

} // namespace keyence
