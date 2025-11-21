#ifndef GUI_MANAGER_H
#define GUI_MANAGER_H

#include "filter_manager.h"
#include <string>

namespace keyence {

/**
 * @brief Manages ImGui sidebar interface
 */
class GUIManager {
public:
    struct GUIState {
        bool filters_changed;
        bool show_filtered_changed;
        bool show_boxes_changed;
        bool export_requested;
        
        GUIState() 
            : filters_changed(false),
              show_filtered_changed(false),
              show_boxes_changed(false),
              export_requested(false) {}
    };
    
    GUIManager();
    
    /**
     * @brief Render the complete sidebar
     * 
     * @param filter_mgr Filter manager with current state
     * @param stats Current statistics
     * @param hover_x Current hover X position (or NaN if invalid)
     * @param hover_y Current hover Y position
     * @param hover_z Current hover Z position
     * @param hover_mode_active Whether hover mode is currently active
     * @param show_filtered Current state of show filtered checkbox
     * @param show_boxes Current state of show boxes checkbox
     * @param sidebar_width Width of sidebar in pixels
     * @param window_height Height of window in pixels
     * @return GUI state indicating what changed
     */
    GUIState render(FilterManager& filter_mgr,
                   const FilterManager::Statistics& stats,
                   float hover_x,
                   float hover_y,
                   float hover_z,
                   bool hover_mode_active,
                   bool& show_filtered,
                   bool& show_boxes,
                   int sidebar_width,
                   int window_height);
    
private:
    // Region dialog state
    bool show_add_region_dialog_;
    RegionalFilter new_region_template_;
    char region_name_buffer_[128];
    int region_to_delete_;
    
    /**
     * @brief Render global filter section
     */
    bool renderGlobalFilterSection(FilterManager& filter_mgr, bool& show_filtered);
    
    /**
     * @brief Render regional filters section
     */
    bool renderRegionalFilterSection(FilterManager& filter_mgr);
    
    /**
     * @brief Render statistics section
     */
    void renderStatisticsSection(const FilterManager::Statistics& stats);
    
    /**
     * @brief Render XYZ location tracker section
     */
    void renderLocationTrackerSection(float x, float y, float z, bool hover_active);
    
    /**
     * @brief Render export section
     */
    bool renderExportSection();
    
    /**
     * @brief Render add region dialog (modal)
     */
    bool renderAddRegionDialog(FilterManager& filter_mgr);
    
    /**
     * @brief Format number with commas
     */
    std::string formatWithCommas(int number);
};

} // namespace keyence

#endif // GUI_MANAGER_H
