#include "keyence_analyzer.h"
#include "filter_manager.h"
#include "color_mapper.h"
#include "vtk_updater.h"
#include "gui_manager.h"

// ImGui includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// VTK includes
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCamera.h>

// GLFW
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>

// Configuration
const int SIDEBAR_WIDTH = 300;
const int INITIAL_WINDOW_WIDTH = 1600;
const int INITIAL_WINDOW_HEIGHT = 900;
const int DEBOUNCE_MS = 200;
const int MAX_DISPLAY_SIZE = 1000;  // Sample to 1000×1000 max

// Global state
keyence::KeyenceAnalyzer* g_analyzer = nullptr;
keyence::FilterManager* g_filter_mgr = nullptr;
keyence::ColorMapper* g_color_mapper = nullptr;
keyence::VTKUpdater* g_vtk_updater = nullptr;
keyence::GUIManager* g_gui_mgr = nullptr;
vtkRenderWindow* g_render_window = nullptr;
vtkRenderWindowInteractor* g_interactor = nullptr;

// Sampled data (what we actually work with)
Eigen::MatrixXf g_sampled_data;
std::vector<int> g_row_indices;
std::vector<int> g_col_indices;

bool g_show_filtered = false;
bool g_show_boxes = true;
bool g_hover_mode_active = false;
float g_hover_x = NAN;
float g_hover_y = NAN;
float g_hover_z = NAN;

bool g_filters_dirty = false;
std::chrono::steady_clock::time_point g_last_change_time;

// Keyboard callback for GLFW
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Let ImGui handle input first
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    
    if (ImGui::GetIO().WantCaptureKeyboard) {
        return;
    }
    
    // Handle 'H' key for hover mode toggle
    if (key == GLFW_KEY_H && action == GLFW_PRESS) {
        g_hover_mode_active = !g_hover_mode_active;
        std::cout << "Hover mode: " << (g_hover_mode_active ? "ON" : "OFF") << std::endl;
        
        if (!g_hover_mode_active) {
            g_hover_x = NAN;
            g_hover_y = NAN;
            g_hover_z = NAN;
        }
    }
}

// Mouse move callback for hover tracking
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (g_hover_mode_active && g_vtk_updater) {
        // Get window size
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        
        // Only pick if mouse is over VTK viewport (right side)
        if (xpos >= SIDEBAR_WIDTH) {
            // Convert to VTK viewport coordinates
            int vtk_x = static_cast<int>(xpos);
            int vtk_y = height - static_cast<int>(ypos);  // Flip Y
            
            g_vtk_updater->pickPoint(vtk_x, vtk_y, g_hover_x, g_hover_y, g_hover_z);
        }
    }
}

void updateFilters() {
    if (!g_analyzer || !g_filter_mgr || !g_color_mapper || !g_vtk_updater) {
        return;
    }
    
    if (g_sampled_data.size() == 0) {
        std::cerr << "Error: No sampled data!" << std::endl;
        return;
    }
    
    std::cout << "Recomputing filters on " << g_sampled_data.rows() 
              << "×" << g_sampled_data.cols() << " sampled points..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    
    // Apply filters to sampled data
    g_filter_mgr->applyFilters(
        g_sampled_data,
        g_analyzer->getPixelPitchX(),
        g_analyzer->getPixelPitchY(),
        g_row_indices,
        g_col_indices
    );
    
    auto after_filter = std::chrono::steady_clock::now();
    
    // Compute colors
    float z_min, z_max;
    g_filter_mgr->getGlobalRange(z_min, z_max);
    
    g_color_mapper->computeColors(
        g_sampled_data,
        g_filter_mgr->getVisibilityMask(),
        g_filter_mgr->getFilterReasons(),
        g_show_filtered,
        z_min,
        z_max
    );
    
    auto after_color = std::chrono::steady_clock::now();
    
    // Update VTK (no additional sampling needed - already sampled)
    g_vtk_updater->updatePointCloud(
        g_sampled_data,
        g_filter_mgr->getVisibilityMask(),
        *g_color_mapper,
        g_show_filtered,
        g_analyzer->getPixelPitchX(),
        g_analyzer->getPixelPitchY(),
        g_row_indices,
        g_col_indices
    );
    
    auto end = std::chrono::steady_clock::now();
    
    auto filter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(after_filter - start).count();
    auto color_ms = std::chrono::duration_cast<std::chrono::milliseconds>(after_color - after_filter).count();
    auto vtk_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - after_color).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Filter: " << filter_ms << "ms, "
              << "Color: " << color_ms << "ms, "
              << "VTK: " << vtk_ms << "ms, "
              << "Total: " << total_ms << "ms" << std::endl;
}

void exportToPLY(const std::string& filename) {
    if (!g_analyzer || !g_filter_mgr) {
        std::cerr << "Cannot export: analyzer or filter manager not initialized" << std::endl;
        return;
    }
    
    std::cout << "Exporting to PLY: " << filename << std::endl;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    const auto& visibility = g_filter_mgr->getVisibilityMask();
    const auto& colors_r = g_color_mapper->getRedChannel();
    const auto& colors_g = g_color_mapper->getGreenChannel();
    const auto& colors_b = g_color_mapper->getBlueChannel();
    
    // Count visible points
    int visible_count = 0;
    for (int i = 0; i < g_sampled_data.rows(); ++i) {
        for (int j = 0; j < g_sampled_data.cols(); ++j) {
            if (!std::isnan(g_sampled_data(i, j)) && visibility(i, j)) {
                visible_count++;
            }
        }
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << visible_count << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Write visible points (using original coordinates)
    float pixel_pitch_x = g_analyzer->getPixelPitchX();
    float pixel_pitch_y = g_analyzer->getPixelPitchY();
    
    for (int i = 0; i < g_sampled_data.rows(); ++i) {
        for (int j = 0; j < g_sampled_data.cols(); ++j) {
            if (!std::isnan(g_sampled_data(i, j)) && visibility(i, j)) {
                // Map back to original coordinates
                int orig_row = g_row_indices[i];
                int orig_col = g_col_indices[j];
                
                float x = orig_col * pixel_pitch_x;
                float y = orig_row * pixel_pitch_y;
                float z = g_sampled_data(i, j) * 1000.0f;
                
                file << x << " " << y << " " << z << " "
                     << static_cast<int>(colors_r(i, j)) << " "
                     << static_cast<int>(colors_g(i, j)) << " "
                     << static_cast<int>(colors_b(i, j)) << "\n";
            }
        }
    }
    
    file.close();
    std::cout << "Exported " << visible_count << " points to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
        return 1;
    }
    
    std::string csv_path = argv[1];
    
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Keyence Analyzer - Interactive Filtering (C++ with GUI)  " << std::endl;
    std::cout << "===========================================================" << std::endl;
    
    // Set locale to avoid issues
    setlocale(LC_ALL, "C");
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
        
    GLFWwindow* window = glfwCreateWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT,
                                         "Keyence Analyzer - Interactive Filtering",
                                         nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync
    
    // Set callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    ImGui::StyleColorsDark();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(nullptr);  // opengl 2.1
    
    // Load data
    std::cout << "\nLoading data..." << std::endl;
    g_analyzer = new keyence::KeyenceAnalyzer();
    auto result = g_analyzer->analyzeSingleFile(csv_path, 500);
    
    if (result.status != "success") {
        std::cerr << "Failed to load file" << std::endl;
        return 1;
    }
    
    std::cout << "Data loaded successfully" << std::endl;
    
    // ==================================================================
    // CRITICAL: SAMPLE FIRST, BEFORE CREATING ANY FILTER STRUCTURES
    // ==================================================================
    
    const auto& full_data = result.raw_data;
    int full_rows = full_data.rows();
    int full_cols = full_data.cols();
    int total_points = full_rows * full_cols;
    
    std::cout << "\nFull dataset: " << full_rows << "×" << full_cols 
              << " = " << total_points << " points" << std::endl;
    
    // Determine sample size
    int sample_rows = std::min(MAX_DISPLAY_SIZE, full_rows);
    int sample_cols = std::min(MAX_DISPLAY_SIZE, full_cols);
    
    std::cout << "Sampling to: " << sample_rows << "×" << sample_cols 
              << " = " << (sample_rows * sample_cols) << " points" << std::endl;
    
    // Create sampling indices
    g_row_indices.clear();
    g_col_indices.clear();
    
    if (sample_rows == full_rows) {
        for (int i = 0; i < full_rows; ++i) g_row_indices.push_back(i);
    } else {
        for (int i = 0; i < sample_rows; ++i) {
            g_row_indices.push_back((i * (full_rows - 1)) / (sample_rows - 1));
        }
    }
    
    if (sample_cols == full_cols) {
        for (int j = 0; j < full_cols; ++j) g_col_indices.push_back(j);
    } else {
        for (int j = 0; j < sample_cols; ++j) {
            g_col_indices.push_back((j * (full_cols - 1)) / (sample_cols - 1));
        }
    }
    
    // Extract sampled data
    g_sampled_data.resize(sample_rows, sample_cols);
    for (int i = 0; i < sample_rows; ++i) {
        for (int j = 0; j < sample_cols; ++j) {
            g_sampled_data(i, j) = full_data(g_row_indices[i], g_col_indices[j]);
        }
    }
    
    std::cout << "Sampled data created: " << g_sampled_data.rows() << "×" 
              << g_sampled_data.cols() << std::endl;
    
    // Compute Z bounds from sampled data
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();
    
    for (int i = 0; i < g_sampled_data.rows(); ++i) {
        for (int j = 0; j < g_sampled_data.cols(); ++j) {
            float val = g_sampled_data(i, j);
            if (!std::isnan(val)) {
                z_min = std::min(z_min, val);
                z_max = std::max(z_max, val);
            }
        }
    }
    
    z_min *= 1000.0f;  // mm to μm
    z_max *= 1000.0f;
    
    std::cout << "Z range: " << z_min << " to " << z_max << " μm" << std::endl;
    
    // Initialize components (now working with sampled data only)
    g_filter_mgr = new keyence::FilterManager();
    g_color_mapper = new keyence::ColorMapper();
    g_vtk_updater = new keyence::VTKUpdater();
    g_gui_mgr = new keyence::GUIManager();
    
    g_filter_mgr->setDataBounds(z_min, z_max);
    g_filter_mgr->setGlobalRange(z_min, z_max);  // Initially show all
    
    // Setup VTK
    g_render_window = vtkRenderWindow::New();
    g_render_window->AddRenderer(g_vtk_updater->getRenderer());
    g_render_window->SetSize(INITIAL_WINDOW_WIDTH - SIDEBAR_WIDTH, INITIAL_WINDOW_HEIGHT);
    g_render_window->SetPosition(SIDEBAR_WIDTH, 0);
    g_render_window->OffScreenRenderingOn();  // ADD THIS LINE

    
    g_interactor = vtkRenderWindowInteractor::New();
    g_interactor->SetRenderWindow(g_render_window);
    
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
        vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    g_interactor->SetInteractorStyle(style);
    
    g_vtk_updater->setRenderWindow(g_render_window);
    
    // Set camera
    vtkCamera* camera = g_vtk_updater->getRenderer()->GetActiveCamera();
    camera->SetPosition(1, -1, 1);
    camera->SetFocalPoint(0, 0, 0);
    camera->SetViewUp(0, 0, 1);
    g_vtk_updater->getRenderer()->ResetCamera();
    
    // Initial visualization
    std::cout << "\nCreating initial visualization..." << std::endl;
    updateFilters();
    
    std::cout << "\nGUI Controls:" << std::endl;
    std::cout << "  - Press 'H' to toggle hover mode (shows XYZ coordinates)" << std::endl;
    std::cout << "  - Use sidebar to adjust filters" << std::endl;
    std::cout << "  - Close window to exit" << std::endl;
    std::cout << "\n===========================================================" << std::endl;
    
    g_last_change_time = std::chrono::steady_clock::now();
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Get window size
        int window_width, window_height;
        glfwGetWindowSize(window, &window_width, &window_height);
        
        // Render GUI
        auto stats = g_filter_mgr->getStatistics();
        auto gui_state = g_gui_mgr->render(
            *g_filter_mgr,
            stats,
            g_hover_x, g_hover_y, g_hover_z,
            g_hover_mode_active,
            g_show_filtered,
            g_show_boxes,
            SIDEBAR_WIDTH,
            window_height
        );
        
        // Handle GUI changes
        if (gui_state.filters_changed || gui_state.show_filtered_changed) {
            g_filters_dirty = true;
            g_last_change_time = std::chrono::steady_clock::now();
        }
        
        if (gui_state.show_boxes_changed) {
            g_vtk_updater->updateRegionBoxes(g_filter_mgr->getRegions(), g_show_boxes);
        }
        
        if (gui_state.export_requested) {
            exportToPLY("filtered_output.ply");
        }
        
        // Debouncing: only update if enough time has passed since last change
        if (g_filters_dirty) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - g_last_change_time).count();
            
            if (elapsed_ms >= DEBOUNCE_MS) {
                updateFilters();
                g_vtk_updater->updateRegionBoxes(g_filter_mgr->getRegions(), g_show_boxes);
                g_filters_dirty = false;
            }
        }
        
        // Render VTK
        g_render_window->Render();
        
        // Render ImGui
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    delete g_gui_mgr;
    delete g_vtk_updater;
    delete g_color_mapper;
    delete g_filter_mgr;
    delete g_analyzer;
    
    g_render_window->Delete();
    g_interactor->Delete();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}