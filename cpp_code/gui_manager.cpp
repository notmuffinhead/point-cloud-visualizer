#include "gui_manager.h"
#include "imgui.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>

namespace keyence {

GUIManager::GUIManager() 
    : show_add_region_dialog_(false),
      region_to_delete_(-1) {
    std::memset(region_name_buffer_, 0, sizeof(region_name_buffer_));
}

GUIManager::GUIState GUIManager::render(FilterManager& filter_mgr,
                                       const FilterManager::Statistics& stats,
                                       float hover_x,
                                       float hover_y,
                                       float hover_z,
                                       bool hover_mode_active,
                                       bool& show_filtered,
                                       bool& show_boxes,
                                       int sidebar_width,
                                       int window_height) {
    GUIState state;
    
    // Create fixed sidebar window
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(sidebar_width), 
                                    static_cast<float>(window_height)));
    
    ImGui::Begin("Controls", nullptr,
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoTitleBar);
    
    ImGui::Text("Keyence Analyzer");
    ImGui::Separator();
    ImGui::Spacing();
    
    // Global Filters Section
    state.filters_changed |= renderGlobalFilterSection(filter_mgr, show_filtered);
    state.show_filtered_changed = state.filters_changed;
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Regional Filters Section
    state.filters_changed |= renderRegionalFilterSection(filter_mgr);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Show boxes checkbox
    if (ImGui::Checkbox("Show Region Outlines", &show_boxes)) {
        state.show_boxes_changed = true;
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Statistics Section
    renderStatisticsSection(stats);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Location Tracker Section
    renderLocationTrackerSection(hover_x, hover_y, hover_z, hover_mode_active);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Export Section
    state.export_requested = renderExportSection();
    
    ImGui::End();
    
    // Handle add region dialog
    if (show_add_region_dialog_) {
        state.filters_changed |= renderAddRegionDialog(filter_mgr);
    }
    
    return state;
}

bool GUIManager::renderGlobalFilterSection(FilterManager& filter_mgr, bool& show_filtered) {
    bool changed = false;
    
    ImGui::TextColored(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), "Global Filters");
    
    float z_min, z_max;
    filter_mgr.getGlobalRange(z_min, z_max);
    
    float z_min_bound, z_max_bound;
    filter_mgr.getDataBounds(z_min_bound, z_max_bound);
    
    // Z-Range slider
    ImGui::Text("Z-Range (μm):");
    
    float range[2] = {z_min, z_max};
    if (ImGui::SliderFloat2("##ZRange", range, z_min_bound, z_max_bound, "%.1f")) {
        filter_mgr.setGlobalRange(range[0], range[1]);
        changed = true;
    }
    
    // Text input boxes for precise values
    ImGui::Text("Min:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("##ZMin", &z_min, 1.0f, 10.0f, "%.1f")) {
        // Clamp and validate
        z_min = std::max(z_min_bound, std::min(z_max_bound, z_min));
        if (z_min > z_max) {
            float temp = z_min;
            z_min = z_max;
            z_max = temp;
        }
        filter_mgr.setGlobalRange(z_min, z_max);
        changed = true;
    }
    
    ImGui::SameLine();
    ImGui::Text("Max:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("##ZMax", &z_max, 1.0f, 10.0f, "%.1f")) {
        // Clamp and validate
        z_max = std::max(z_min_bound, std::min(z_max_bound, z_max));
        if (z_max < z_min) {
            float temp = z_max;
            z_max = z_min;
            z_min = temp;
        }
        filter_mgr.setGlobalRange(z_min, z_max);
        changed = true;
    }
    
    ImGui::Spacing();
    
    // Show filtered checkbox
    if (ImGui::Checkbox("Show Filtered Points", &show_filtered)) {
        changed = true;
    }
    
    return changed;
}

bool GUIManager::renderRegionalFilterSection(FilterManager& filter_mgr) {
    bool changed = false;
    
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Regional Filters");
    
    const auto& regions = filter_mgr.getRegions();
    
    float z_min_bound, z_max_bound;
    filter_mgr.getDataBounds(z_min_bound, z_max_bound);
    
    // Display existing regions
    for (const auto& region : regions) {
        ImGui::PushID(region.id);
        
        if (ImGui::TreeNode(region.name.c_str())) {
            RegionalFilter modified = region;
            
            // Active checkbox
            if (ImGui::Checkbox("Active", &modified.active)) {
                filter_mgr.updateRegion(region.id, modified);
                changed = true;
            }
            
            // X Range
            ImGui::Text("X Range (μm):");
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("##XMin", &modified.x_min, 10.0f, 100.0f, "%.1f");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("##XMax", &modified.x_max, 10.0f, 100.0f, "%.1f")) {
                filter_mgr.updateRegion(region.id, modified);
                changed = true;
            }
            
            // Y Range
            ImGui::Text("Y Range (μm):");
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("##YMin", &modified.y_min, 10.0f, 100.0f, "%.1f");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("##YMax", &modified.y_max, 10.0f, 100.0f, "%.1f")) {
                filter_mgr.updateRegion(region.id, modified);
                changed = true;
            }
            
            // Z Range slider
            ImGui::Text("Z Range (μm):");
            float z_range[2] = {modified.z_min, modified.z_max};
            if (ImGui::SliderFloat2("##RegionZ", z_range, z_min_bound, z_max_bound, "%.1f")) {
                modified.z_min = z_range[0];
                modified.z_max = z_range[1];
                filter_mgr.updateRegion(region.id, modified);
                changed = true;
            }
            
            ImGui::Spacing();
            
            // Delete button
            if (ImGui::Button("Delete Region")) {
                region_to_delete_ = region.id;
                changed = true;
            }
            
            ImGui::TreePop();
        }
        
        ImGui::PopID();
    }
    
    // Handle deletion (do after iteration to avoid iterator invalidation)
    if (region_to_delete_ >= 0) {
        filter_mgr.removeRegion(region_to_delete_);
        region_to_delete_ = -1;
    }
    
    ImGui::Spacing();
    
    // Add region button
    if (ImGui::Button("+ Add New Region")) {
        show_add_region_dialog_ = true;
        
        // Initialize template with reasonable defaults
        new_region_template_ = RegionalFilter();
        new_region_template_.name = "New Region";
        std::strcpy(region_name_buffer_, "New Region");
        
        // Set default bounds to middle 50% of data
        float z_min_bound, z_max_bound;
        filter_mgr.getDataBounds(z_min_bound, z_max_bound);
        float z_range = z_max_bound - z_min_bound;
        new_region_template_.z_min = z_min_bound + z_range * 0.25f;
        new_region_template_.z_max = z_max_bound - z_range * 0.25f;
    }
    
    return changed;
}

bool GUIManager::renderAddRegionDialog(FilterManager& filter_mgr) {
    bool changed = false;
    
    ImGui::OpenPopup("Add Region");
    
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    
    if (ImGui::BeginPopupModal("Add Region", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Define a new regional filter:");
        ImGui::Separator();
        ImGui::Spacing();
        
        ImGui::Text("Name:");
        ImGui::InputText("##RegionName", region_name_buffer_, sizeof(region_name_buffer_));
        
        ImGui::Spacing();
        
        ImGui::Text("X Range (μm):");
        ImGui::InputFloat("##NewXMin", &new_region_template_.x_min, 10.0f, 100.0f, "%.1f");
        ImGui::InputFloat("##NewXMax", &new_region_template_.x_max, 10.0f, 100.0f, "%.1f");
        
        ImGui::Spacing();
        
        ImGui::Text("Y Range (μm):");
        ImGui::InputFloat("##NewYMin", &new_region_template_.y_min, 10.0f, 100.0f, "%.1f");
        ImGui::InputFloat("##NewYMax", &new_region_template_.y_max, 10.0f, 100.0f, "%.1f");
        
        ImGui::Spacing();
        
        float z_min_bound, z_max_bound;
        filter_mgr.getDataBounds(z_min_bound, z_max_bound);
        
        ImGui::Text("Z Range (μm):");
        float z_range[2] = {new_region_template_.z_min, new_region_template_.z_max};
        ImGui::SliderFloat2("##NewZRange", z_range, z_min_bound, z_max_bound, "%.1f");
        new_region_template_.z_min = z_range[0];
        new_region_template_.z_max = z_range[1];
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Buttons
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            new_region_template_.name = region_name_buffer_;
            filter_mgr.addRegion(new_region_template_);
            show_add_region_dialog_ = false;
            changed = true;
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_add_region_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::EndPopup();
    }
    
    return changed;
}

void GUIManager::renderStatisticsSection(const FilterManager::Statistics& stats) {
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Statistics");
    
    ImGui::Text("Total Points: %s", formatWithCommas(stats.total_points).c_str());
    
    ImGui::Spacing();
    
    float visible_pct = stats.total_points > 0 ? 
        100.0f * stats.visible_count / stats.total_points : 0.0f;
    float global_pct = stats.total_points > 0 ?
        100.0f * stats.global_filtered_count / stats.total_points : 0.0f;
    float regional_pct = stats.total_points > 0 ?
        100.0f * stats.regional_filtered_count / stats.total_points : 0.0f;
    
    ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "Visible:");
    ImGui::SameLine();
    ImGui::Text("%s (%.1f%%)", formatWithCommas(stats.visible_count).c_str(), visible_pct);
    
    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Filtered (Global):");
    ImGui::SameLine();
    ImGui::Text("%s (%.1f%%)", formatWithCommas(stats.global_filtered_count).c_str(), global_pct);
    
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Filtered (Regional):");
    ImGui::SameLine();
    ImGui::Text("%s (%.1f%%)", formatWithCommas(stats.regional_filtered_count).c_str(), regional_pct);
}

void GUIManager::renderLocationTrackerSection(float x, float y, float z, bool hover_active) {
    ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.8f, 1.0f), "XYZ Location");
    
    if (hover_active) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Status: ON");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Status: OFF");
    }
    
    ImGui::Text("(Press 'H' to toggle)");
    
    ImGui::Spacing();
    
    if (hover_active && !std::isnan(x)) {
        ImGui::Text("X: %.1f μm", x);
        ImGui::Text("Y: %.1f μm", y);
        ImGui::Text("Z: %.1f μm", z);
    } else {
        ImGui::TextDisabled("X: ---");
        ImGui::TextDisabled("Y: ---");
        ImGui::TextDisabled("Z: ---");
    }
}

bool GUIManager::renderExportSection() {
    if (ImGui::Button("Export to PLY", ImVec2(-1, 30))) {
        return true;
    }
    return false;
}

std::string GUIManager::formatWithCommas(int number) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << number;
    return ss.str();
}

} // namespace keyence
