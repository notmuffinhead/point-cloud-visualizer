# Architecture Diagram: Interactive Filtering GUI

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌────────────────────┬─────────────────────────────────────┐   │
│  │  ImGui Sidebar     │     VTK 3D Viewport                │   │
│  │  (gui_manager)     │     (vtk_updater)                  │   │
│  │                    │                                    │   │
│  │  • Global Filter   │     [3D Point Cloud Visualization] │   │
│  │  • Regional Filter │                                    │   │
│  │  • Statistics      │     [Wireframe Boxes for Regions]  │   │
│  │  • XYZ Tracker     │                                    │   │
│  │  • Export Button   │     [Interactive Camera Controls]  │   │
│  └────────────────────┴─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ User Input
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LOGIC                           │
│                       (main_gui.cpp)                            │
│                                                                 │
│  • GLFW Window Management                                       │
│  • ImGui Frame Loop                                             │
│  • Debouncing Logic                                             │
│  • Keyboard Input ('H' for hover)                              │
│  • Coordinates Component Updates                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ FilterManager  │ │  ColorMapper   │ │   VTKUpdater   │
        └────────────────┘ └────────────────┘ └────────────────┘
                │                 │                     │
                ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                  (keyence_analyzer.h/cpp)                       │
│                                                                 │
│  • Eigen::MatrixXf (Height Data)                               │
│  • CSV Loading                                                  │
│  • Coordinate Transformations                                   │
│  • [COMPLETELY UNCHANGED FROM YOUR ORIGINAL CODE]               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### 1. Initial Load

```
main_gui.cpp
    │
    ├─► Load CSV via KeyenceAnalyzer
    │   └─► Eigen::MatrixXf heights
    │
    ├─► Compute Z bounds from data
    │   └─► z_min = heights.minCoeff()
    │   └─► z_max = heights.maxCoeff()
    │
    ├─► Initialize FilterManager
    │   └─► setDataBounds(z_min, z_max)
    │   └─► setGlobalRange(z_min, z_max)  // Show all initially
    │
    ├─► Initialize ColorMapper
    │   └─► Create VTK colormap (blue→red)
    │
    ├─► Initialize VTKUpdater
    │   └─► Create VTK renderer
    │
    └─► Initialize GUIManager
        └─► Setup ImGui context
```

### 2. User Adjusts Global Z-Range Slider

```
User drags slider
    │
    ▼
gui_manager.cpp
    │
    ├─► ImGui::SliderFloat2() detects change
    ├─► filter_mgr.setGlobalRange(new_min, new_max)
    └─► Return: gui_state.filters_changed = true
    │
    ▼
main_gui.cpp
    │
    ├─► Detect filters_changed
    ├─► Mark g_filters_dirty = true
    ├─► Record g_last_change_time = now
    │
    ▼
[Wait for debounce period - 200ms]
    │
    ▼
main_gui.cpp → updateFilters()
    │
    ├─► FilterManager::applyFilters()
    │   │
    │   └─► For each point (row, col):
    │       ├─► Get physical coords (x, y, z)
    │       ├─► Check if in any region
    │       ├─► Apply appropriate Z-range test
    │       └─► Set visibility_mask(row, col) = true/false
    │
    ├─► ColorMapper::computeColors()
    │   │
    │   └─► For each point (row, col):
    │       ├─► If visible: normalized = (z - z_min) / (z_max - z_min)
    │       │   └─► color = colormap(normalized)
    │       └─► If filtered & show_filtered: color = red/orange
    │
    └─► VTKUpdater::updatePointCloud()
        │
        └─► For each point (row, col):
            ├─► If should_render(visibility, show_filtered):
            │   └─► Add to vtkPoints with assigned color
            └─► Create quads between neighboring points
    │
    ▼
VTK renders updated scene
```

### 3. User Adds Regional Filter

```
User clicks "+ Add New Region"
    │
    ▼
gui_manager.cpp
    │
    ├─► Open modal dialog
    ├─► User enters X, Y, Z bounds
    ├─► User clicks OK
    │
    ├─► filter_mgr.addRegion(new_region)
    │   └─► regions_.push_back(new_region)
    │
    └─► Return: gui_state.filters_changed = true
    │
    ▼
[Same update flow as slider adjustment]
    │
    ├─► applyFilters() now checks regional filters
    ├─► Points in region use region's Z-range
    ├─► Points outside region use global Z-range
    │
    └─► VTKUpdater also creates wireframe box
        └─► vtkCubeSource at region bounds
```

### 4. User Presses 'H' Key

```
User presses 'H'
    │
    ▼
main_gui.cpp → keyCallback()
    │
    └─► g_hover_mode_active = !g_hover_mode_active
    │
    ▼
[If hover mode ON]
    │
    └─► On every mouse move:
        │
        ├─► cursorPosCallback()
        │   └─► vtk_updater.pickPoint(screen_x, screen_y)
        │       ├─► VTK ray-casting to find 3D point
        │       └─► Return world coordinates (x, y, z)
        │
        └─► gui_manager.render() displays coordinates
            └─► ImGui::Text("X: %.1f μm", hover_x)
```

### 5. User Clicks Export

```
User clicks "Export to PLY"
    │
    ▼
gui_manager.cpp
    │
    └─► Return: gui_state.export_requested = true
    │
    ▼
main_gui.cpp → exportToPLY()
    │
    ├─► Create PLY file
    ├─► Write header with vertex count
    │
    └─► For each point (row, col):
        ├─► If visibility_mask(row, col) == true:
        │   ├─► Calculate physical coords
        │   └─► Write: x y z r g b
        └─► Skip filtered points
    │
    ▼
File saved: filtered_output.ply
```

## Data Structures

### FilterManager

```
┌─────────────────────────────────────┐
│         FilterManager               │
├─────────────────────────────────────┤
│ State:                              │
│  • global_z_min_value_  (float)     │
│  • global_z_max_value_  (float)     │
│  • global_z_min_bound_  (float)     │
│  • global_z_max_bound_  (float)     │
│  • regions_  (vector)               │
│                                     │
│ Output:                             │
│  • visibility_mask_  (bool matrix)  │
│  • filter_reason_    (uint8 matrix) │
└─────────────────────────────────────┘
```

### ColorMapper

```
┌──────────────────────────────────────┐
│          ColorMapper                 │
├──────────────────────────────────────┤
│ Colormap:                            │
│  • vtkColorTransferFunction          │
│    (blue → red gradient)             │
│                                      │
│ Output:                              │
│  • colors_r_  (uint8 matrix)         │
│  • colors_g_  (uint8 matrix)         │
│  • colors_b_  (uint8 matrix)         │
└──────────────────────────────────────┘
```

### VTKUpdater

```
┌──────────────────────────────────────┐
│           VTKUpdater                 │
├──────────────────────────────────────┤
│ VTK Objects:                         │
│  • vtkRenderer                       │
│  • vtkPolyData                       │
│  • vtkPoints                         │
│  • vtkActor (surface)                │
│  • vtkActor[] (region boxes)         │
│  • vtkCellPicker                     │
└──────────────────────────────────────┘
```

## Memory Layout (3000×3000 example)

```
┌────────────────────────────────────────┐
│  Eigen::MatrixXf heights               │  36 MB
│  (from KeyenceAnalyzer)                │  [EXISTING]
└────────────────────────────────────────┘
              │
              │ Read by FilterManager
              ▼
┌────────────────────────────────────────┐
│  Eigen::Matrix<bool> visibility_mask   │   9 MB
│  (FilterManager output)                │  [NEW]
└────────────────────────────────────────┘
              │
              │ Read by ColorMapper
              ▼
┌────────────────────────────────────────┐
│  Eigen::Matrix<uint8> colors_r/g/b     │  27 MB
│  (ColorMapper output)                  │  [NEW]
└────────────────────────────────────────┘
              │
              │ Read by VTKUpdater
              ▼
┌────────────────────────────────────────┐
│  vtkPoints + vtkPolyData               │ ~15 MB
│  (VTK rendering structures)            │  [NEW]
└────────────────────────────────────────┘
```

## Timing Breakdown (Single-threaded, 9M points)

```
User stops dragging slider
    │
    ├─► [0ms] Debounce wait (200ms)
    │
    ├─► [200ms] Start updateFilters()
    │   │
    │   ├─► FilterManager::applyFilters()
    │   │   • Loop through 9M points
    │   │   • Check Z-range & regions
    │   │   • Update visibility mask
    │   │   └─► ~50ms
    │   │
    │   ├─► ColorMapper::computeColors()
    │   │   • Loop through 9M points
    │   │   • Normalize & colormap lookup
    │   │   • Assign RGB values
    │   │   └─► ~30ms
    │   │
    │   └─► VTKUpdater::updatePointCloud()
    │       • Build vtkPoints array
    │       • Create quad mesh
    │       • Update actor
    │       └─► ~150ms
    │
    └─► [430ms] Display updated visualization
```

**Total perceived delay: ~430ms from slider release**

## Thread Safety

**Current Implementation: Single-threaded**
- All updates happen on main thread
- No race conditions
- No mutex/locking needed
- Simpler code, easier debugging

**If parallelization needed later:**
```
Main Thread:
    • GUI rendering (60 FPS)
    • VTK rendering (as fast as possible)

Background Thread:
    • FilterManager::applyFilters()
    • ColorMapper::computeColors()
    • VTKUpdater::updatePointCloud()
    
Synchronization:
    • Mutex around data swap
    • Main thread shows old data while background computes new
```

## File Organization

```
your_project/
│
├─── Data Layer (Existing) ───────────┐
│    • keyence_analyzer.h/cpp         │ UNCHANGED
│                                      │
├─── Filter Logic (New) ──────────────┤
│    • filter_manager.h/cpp           │ Filtering
│    • color_mapper.h/cpp             │ Coloring
│                                      │
├─── Visualization (New) ─────────────┤
│    • vtk_updater.h/cpp              │ VTK rendering
│    • gui_manager.h/cpp              │ ImGui sidebar
│                                      │
├─── Application (New) ───────────────┤
│    • main_gui.cpp                   │ Main loop
│    • CMakeLists.txt                 │ Build system
│                                      │
└─── External Dependencies ───────────┘
     • imgui/ (download)
     • GLFW (install)
     • OpenGL (system)
```

## Summary

**Layers:**
1. **UI Layer** (ImGui + VTK): User interaction and visualization
2. **Logic Layer** (FilterManager, ColorMapper): Computation
3. **Rendering Layer** (VTKUpdater): Graphics pipeline
4. **Data Layer** (KeyenceAnalyzer): Storage (unchanged)

**Data Flow:**
- **One-way**: Data flows down (no circular dependencies)
- **Event-driven**: Updates triggered by user input only
- **Debounced**: Changes batched to prevent lag

**Performance:**
- **Single-threaded**: Simple, adequate for 9M points
- **Optimized**: Compiler flags, early exits, efficient loops
- **Responsive**: Debouncing makes UI feel instant

**Architecture:**
- **Modular**: Each class has one clear responsibility
- **Testable**: Components can be tested independently
- **Extensible**: Easy to add new filter types or visualizations
- **Non-invasive**: Your original code completely unchanged
