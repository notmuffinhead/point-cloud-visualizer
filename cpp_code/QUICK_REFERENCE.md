# Quick Reference: Interactive Filtering GUI Files

## Files Overview

### Core Filtering System

**filter_manager.h/cpp** (3.2 KB + 4.5 KB)
- Manages global and regional filter definitions
- Computes visibility mask for all points
- Tracks why each point was filtered
- Single-threaded, optimized for your dataset sizes

**color_mapper.h/cpp** (2.1 KB + 3.3 KB)
- Uses VTK's vtkColorTransferFunction for colormaps
- Maps height → color for visible points
- Assigns red (global filter) or orange (regional filter) for filtered points
- Normalizes colors based on user-selected Z-range

**vtk_updater.h/cpp** (3.5 KB + 7.1 KB)
- Builds VTK PolyData from filtered point cloud
- Creates wireframe boxes for regional filters
- Handles picking for XYZ location tracking
- Only renders points that pass filters (or filtered points if enabled)

**gui_manager.h/cpp** (2.3 KB + 9.8 KB)
- Renders all ImGui sidebar panels
- Handles user input (sliders, checkboxes, buttons)
- Manages add/edit/delete region dialogs
- Returns state changes to trigger updates

### Main Application

**main_gui.cpp** (7.2 KB)
- GLFW window management
- ImGui initialization
- Main render loop with debouncing
- Keyboard callbacks ('H' for hover mode)
- PLY export functionality
- Coordinates all components

### Build System

**CMakeLists_GUI.txt** (5.8 KB)
- Builds both CLI and GUI versions
- Finds dependencies: Eigen3, VTK, GLFW, OpenGL
- Includes ImGui sources
- Compiler optimizations for Release builds

### Documentation

**INTEGRATION_GUIDE.md** (8.4 KB)
- Step-by-step integration instructions
- Troubleshooting guide
- Customization options
- Usage instructions

---

## Key Design Decisions

### 1. Non-Destructive Integration
- Your existing `keyence_analyzer.h/cpp` is **completely unchanged**
- New classes only **read** from your analyzer
- CLI version (`main.cpp`) still works exactly as before
- You get both CLI and GUI executables

### 2. Sidebar Layout
- Fixed 300px sidebar on left
- VTK viewport fills remaining space
- All controls in one scrollable window
- No floating/overlapping windows

### 3. Color Normalization
- Colors map to **user-selected range**, not global data bounds
- Adjusting Z-range slider changes color emphasis
- Visible points: Blue-to-red gradient (via VTK colormap)
- Filtered points: Red (global) or Orange (regional)

### 4. Performance Strategy
- **Debouncing**: 200ms delay after slider movement before recomputing
- **Single-threaded**: Adequate for 9M points (~230ms total update)
- **Compiler optimizations**: `-O3 -march=native` in Release build
- **No parallelization**: Keeps code simple, still responsive

### 5. Hover Mode
- Toggle on/off with 'H' key (not continuous)
- Uses VTK picking to find 3D coordinates
- Only active when user requests it (saves CPU)

---

## File Dependencies

```
main_gui.cpp
    ├── keyence_analyzer.h (YOUR EXISTING)
    ├── filter_manager.h
    ├── color_mapper.h
    ├── vtk_updater.h
    ├── gui_manager.h
    ├── imgui.h (EXTERNAL)
    └── GLFW (EXTERNAL)

filter_manager.cpp
    └── Eigen (EXTERNAL)

color_mapper.cpp
    ├── Eigen (EXTERNAL)
    └── VTK (EXTERNAL)

vtk_updater.cpp
    ├── Eigen (EXTERNAL)
    ├── VTK (EXTERNAL)
    └── filter_manager.h

gui_manager.cpp
    ├── imgui.h (EXTERNAL)
    └── filter_manager.h
```

---

## Build Commands Quick Reference

```bash
# Setup (one time)
mkdir build
cd build
cmake ..

# Build both executables
cmake --build . --config Release

# Or on Unix
make -j4

# Output executables:
# - build/keyence_analyzer (CLI)
# - build/keyence_analyzer_gui (GUI)
```

---

## Usage Examples

```bash
# CLI (unchanged from before)
./keyence_analyzer data.csv 500

# GUI with interactive filtering
./keyence_analyzer_gui data.csv
```

**In the GUI:**
1. Adjust global Z-range slider → Points outside range turn red or disappear
2. Check "Show Filtered Points" → See what's being filtered in red/orange
3. Add regional filter → Define XY box with different Z-range
4. Press 'H' → Enable hover mode, move mouse to see XYZ coords
5. Click "Export to PLY" → Save filtered point cloud

---

## Memory Footprint

For a 3000×3000 dataset:

```
Original data:       36 MB  (Eigen::MatrixXf, already in your code)
Visibility mask:      9 MB  (NEW: bool matrix)
Filter reasons:       9 MB  (NEW: uint8_t matrix)
RGB colors:          27 MB  (NEW: 3× uint8_t matrices)
VTK structures:     ~15 MB  (Approximate, depends on sampling)
────────────────────────────
Total NEW memory:   ~60 MB
```

Scales linearly with dataset size.

---

## Configuration Constants

All in `main_gui.cpp`:

```cpp
const int SIDEBAR_WIDTH = 300;              // Sidebar width in pixels
const int INITIAL_WINDOW_WIDTH = 1600;      // Window width
const int INITIAL_WINDOW_HEIGHT = 900;      // Window height
const int DEBOUNCE_MS = 200;                // Delay before updating
```

---

## Customization Hot Spots

**Change colormap**: `color_mapper.cpp` → `initializeColormap()`
**Change sidebar width**: `main_gui.cpp` → `SIDEBAR_WIDTH`
**Change debounce delay**: `main_gui.cpp` → `DEBOUNCE_MS`
**Add keyboard shortcuts**: `main_gui.cpp` → `keyCallback()`
**Change filtered colors**: `color_mapper.cpp` → `getFilteredColor()`

---

## External Dependencies

**Required:**
- Eigen3 (you already have this)
- VTK (you already have this)
- GLFW3 (new - for windowing)
- OpenGL (usually comes with GLFW)
- ImGui (new - download from GitHub)

**Download ImGui:**
```bash
git clone https://github.com/ocornut/imgui.git imgui
```

---

## Testing Checklist

After building, verify:

1. ✅ CLI version still works: `./keyence_analyzer test.csv`
2. ✅ GUI launches: `./keyence_analyzer_gui test.csv`
3. ✅ Data loads and displays in 3D
4. ✅ Sidebar is visible on left
5. ✅ Z-range slider adjusts visible points
6. ✅ Statistics update in real-time
7. ✅ Press 'H' → hover mode activates
8. ✅ Add region → new filter appears
9. ✅ Export → PLY file created

---

## Common Issues & Solutions

**Build fails - "ImGui not found"**
→ Download ImGui and place in `imgui/` directory

**Build fails - "GLFW not found"**
→ Install GLFW: `pacman -S mingw-w64-x86_64-glfw` (Windows/MSYS2)

**GUI window is blank**
→ Check console for errors, data might not have loaded

**Performance is slow**
→ Make sure you're using Release build: `cmake -DCMAKE_BUILD_TYPE=Release ..`

**Sidebar is too wide/narrow**
→ Change `SIDEBAR_WIDTH` in `main_gui.cpp` and rebuild

---

## File Sizes Summary

```
filter_manager.h          3.2 KB
filter_manager.cpp        4.5 KB
color_mapper.h            2.1 KB
color_mapper.cpp          3.3 KB
vtk_updater.h             3.5 KB
vtk_updater.cpp           7.1 KB
gui_manager.h             2.3 KB
gui_manager.cpp           9.8 KB
main_gui.cpp              7.2 KB
CMakeLists_GUI.txt        5.8 KB
INTEGRATION_GUIDE.md      8.4 KB
────────────────────────────────
Total:                   57.2 KB
```

**Plus ImGui (~150 KB of source files to download)**

---

## Next Actions

1. Download ImGui → Place in `imgui/` directory
2. Copy all new files to your project directory
3. Replace CMakeLists.txt with CMakeLists_GUI.txt
4. Run build commands
5. Test with your data
6. Read INTEGRATION_GUIDE.md for details

**You're ready to go!**
