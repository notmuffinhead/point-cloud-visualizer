cmake_minimum_required(VERSION 3.10)
project(KeyenceAnalyzerGUI VERSION 1.0 LANGUAGES CXX)

# ═══════════════════════════════════════════════════════════════════
# C++ Standard
# ═══════════════════════════════════════════════════════════════════
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ═══════════════════════════════════════════════════════════════════
# Build Type
# ═══════════════════════════════════════════════════════════════════
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ═══════════════════════════════════════════════════════════════════
# Options
# ═══════════════════════════════════════════════════════════════════
option(BUILD_GUI "Build GUI application with ImGui" ON)

# ═══════════════════════════════════════════════════════════════════
# Find Required Packages
# ═══════════════════════════════════════════════════════════════════

# Eigen3 (for numerical operations)
find_package(Eigen3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3 version: ${EIGEN3_VERSION}")


# VTK (for visualization) - Compatible with VTK 7.1
find_package(VTK REQUIRED)

if(VTK_FOUND)
    message(STATUS "Found VTK version: ${VTK_VERSION}")
    
    if(VTK_VERSION VERSION_GREATER_EQUAL "9.0")
        message(STATUS "Using VTK 9+ component system")
    else()
        message(STATUS "Using VTK 7/8 compatibility mode")
        include(${VTK_USE_FILE})
    endif()
else()
    message(FATAL_ERROR "VTK not found.")
endif()


# ═══════════════════════════════════════════════════════════════════
# Include Directories
# ═══════════════════════════════════════════════════════════════════
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# ═══════════════════════════════════════════════════════════════════
# Core Library Sources (existing analyzer)
# ═══════════════════════════════════════════════════════════════════
set(CORE_SOURCES
    keyence_analyzer.cpp
)

set(CORE_HEADERS
    keyence_analyzer.h
)

# Create core library
add_library(keyence_core ${CORE_SOURCES} ${CORE_HEADERS})
target_link_libraries(keyence_core
    ${VTK_LIBRARIES}
    Eigen3::Eigen
)

# ═══════════════════════════════════════════════════════════════════
# CLI Executable (existing)
# ═══════════════════════════════════════════════════════════════════
add_executable(keyence_analyzer main.cpp)
target_link_libraries(keyence_analyzer keyence_core)

# VTK Module Initialization
if(VTK_VERSION VERSION_GREATER_EQUAL "9.0")
    vtk_module_autoinit(
        TARGETS keyence_analyzer
        MODULES ${VTK_LIBRARIES}
    )
endif()

# ═══════════════════════════════════════════════════════════════════
# GUI Application (new)
# ═══════════════════════════════════════════════════════════════════
if(BUILD_GUI)
    message(STATUS "Building GUI application")
    
    # Find GLFW
    find_package(glfw3 REQUIRED)
    message(STATUS "Found GLFW3")
    
    # Find OpenGL
    find_package(OpenGL REQUIRED)
    message(STATUS "Found OpenGL")
    
    # ImGui sources (assuming ImGui is in imgui/ subdirectory)
    set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
    
    if(NOT EXISTS "${IMGUI_DIR}/imgui.h")
        message(FATAL_ERROR "ImGui not found in ${IMGUI_DIR}. Please download ImGui and place it in the imgui/ directory.")
    endif()
    
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    
    # GUI component sources
    set(GUI_SOURCES
        filter_manager.cpp
        color_mapper.cpp
        vtk_updater.cpp
        gui_manager.cpp
        main_gui.cpp
        ${IMGUI_SOURCES}
    )
    
    set(GUI_HEADERS
        filter_manager.h
        color_mapper.h
        vtk_updater.h
        gui_manager.h
    )
    
    # Create GUI executable
    add_executable(keyence_analyzer_gui ${GUI_SOURCES} ${GUI_HEADERS})
    
    # Include ImGui directories
    target_include_directories(keyence_analyzer_gui PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
    )
    
    # Link libraries
    target_link_libraries(keyence_analyzer_gui
        keyence_core
        glfw
        OpenGL::GL
    )
    
    # VTK Module Initialization for GUI
    if(VTK_VERSION VERSION_GREATER_EQUAL "9.0")
        vtk_module_autoinit(
            TARGETS keyence_analyzer_gui
            MODULES ${VTK_LIBRARIES}
        )
    endif()
    
    message(STATUS "GUI executable: keyence_analyzer_gui")
endif()

# ═══════════════════════════════════════════════════════════════════
# Compiler Warnings
# ═══════════════════════════════════════════════════════════════════
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(keyence_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
elseif(MSVC)
    target_compile_options(keyence_core PRIVATE
        /W4
    )
endif()

# Compiler optimizations for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(keyence_core PRIVATE
            -O3
            -march=native
        )
    endif()
endif()

# ═══════════════════════════════════════════════════════════════════
# Installation
# ═══════════════════════════════════════════════════════════════════
install(TARGETS keyence_analyzer
        RUNTIME DESTINATION bin)

if(BUILD_GUI)
    install(TARGETS keyence_analyzer_gui
            RUNTIME DESTINATION bin)
endif()

install(FILES ${CORE_HEADERS}
        DESTINATION include/keyence_analyzer)

# ═══════════════════════════════════════════════════════════════════
# Build Information Summary
# ═══════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "Keyence Analyzer Configuration Summary:")
message(STATUS "  C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:           ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Eigen3 Version:     ${EIGEN3_VERSION}")
message(STATUS "  VTK Version:        ${VTK_VERSION}")
message(STATUS "  Build GUI:          ${BUILD_GUI}")
if(BUILD_GUI)
    message(STATUS "  GLFW Found:         YES")
    message(STATUS "  OpenGL Found:       YES")
endif()
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "")
