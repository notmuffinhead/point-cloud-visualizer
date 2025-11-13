cmake_minimum_required(VERSION 3.10)
project(KeyenceAnalyzer VERSION 1.0 LANGUAGES CXX)

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
# Find Required Packages
# ═══════════════════════════════════════════════════════════════════

# Eigen3 (for numerical operations)
find_package(Eigen3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3 version: ${EIGEN3_VERSION}")

# VTK (for visualization)
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    FiltersCore
    FiltersGeometry
    RenderingCore
    RenderingOpenGL2
    InteractionStyle
    InteractionWidgets
    RenderingAnnotation
    IOImage
)

if(VTK_FOUND)
    message(STATUS "Found VTK version: ${VTK_VERSION}")
    include(${VTK_USE_FILE})
else()
    message(FATAL_ERROR "VTK not found. Please install VTK.")
endif()

# ═══════════════════════════════════════════════════════════════════
# Include Directories
# ═══════════════════════════════════════════════════════════════════
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# ═══════════════════════════════════════════════════════════════════
# Source Files
# ═══════════════════════════════════════════════════════════════════
set(SOURCES
    keyence_analyzer.cpp
    main.cpp
)

set(HEADERS
    keyence_analyzer.h
)

# ═══════════════════════════════════════════════════════════════════
# Create Executable
# ═══════════════════════════════════════════════════════════════════
add_executable(keyence_analyzer ${SOURCES} ${HEADERS})

# ═══════════════════════════════════════════════════════════════════
# Link Libraries
# ═══════════════════════════════════════════════════════════════════
target_link_libraries(keyence_analyzer
    ${VTK_LIBRARIES}
    Eigen3::Eigen
)

# Link filesystem library (needed for GCC < 9)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(keyence_analyzer stdc++fs)
endif()


# ═══════════════════════════════════════════════════════════════════
# VTK Module Initialization (for VTK 9.0+)
# ═══════════════════════════════════════════════════════════════════
if(VTK_VERSION VERSION_GREATER_EQUAL "9.0")
    vtk_module_autoinit(
        TARGETS keyence_analyzer
        MODULES ${VTK_LIBRARIES}
    )
endif()

# ═══════════════════════════════════════════════════════════════════
# Optional: OpenMP for Parallel Processing
# ═══════════════════════════════════════════════════════════════════
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(keyence_analyzer OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - parallel processing enabled")
else()
    message(STATUS "OpenMP not found - continuing without parallel support")
endif()

# ═══════════════════════════════════════════════════════════════════
# Compiler Warnings
# ═══════════════════════════════════════════════════════════════════
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(keyence_analyzer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
elseif(MSVC)
    target_compile_options(keyence_analyzer PRIVATE
        /W4
    )
endif()

# ═══════════════════════════════════════════════════════════════════
# Installation
# ═══════════════════════════════════════════════════════════════════
install(TARGETS keyence_analyzer 
        RUNTIME DESTINATION bin)

install(FILES ${HEADERS}
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
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP:             Enabled")
else()
    message(STATUS "  OpenMP:             Disabled")
endif()
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "")
