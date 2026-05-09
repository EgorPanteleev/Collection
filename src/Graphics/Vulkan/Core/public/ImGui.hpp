//
// Created by igor on 4/23/26.
//

#ifndef COLLECTION_IMGUI_HPP
#define COLLECTION_IMGUI_HPP

#include <imgui.h>

#include "Context.hpp"
#include "DescriptorPool.hpp"
#include "Swapchain.hpp"

namespace crv::graphics::vulkan {
    struct ImGuiCreateInfo {
        Context*   context    = nullptr;
        uint32_t   imageCount = 1;
        VkFormat   format     = VK_FORMAT_UNDEFINED;
        float      alpha = 1.0f;
        float      scale = 1.0f;
    };

    struct ImGuiRenderInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkImageView     imageView = VK_NULL_HANDLE;
        VkExtent2D      extent{};
    };

    struct ImGuiTab {
        const char* label;
        uint32_t index;
        std::function<void()> func;
    };
    using TabPanel = std::vector<ImGuiTab>;

    class VkImGui {
    public:
        VkImGui() = default;
        explicit VkImGui(const ImGuiCreateInfo& info);
        VkImGui(VkImGui&&);
        ~VkImGui();
        VkImGui& operator=(VkImGui&&) noexcept;
        [[nodiscard]] ImDrawData* drawData() { return mDrawData; }
        void destroy();
        void beginFrame();
        void endFrame();
        void render(const ImGuiRenderInfo& info);
        static void loadConfigFile(const char* path);
        static void saveConfigFile(const char* path);
        static bool selectableButton(const char* label, bool cond);
        static void beginGroup(const char* name);
        static void endGroup();
        static void separatorText(const char* label);
        static bool beginCompactTable(const char* tableId, float padding);
        static void endCompactTable();
        static void row(const char* label, const char* value);
        static bool tabButton(const char* label, bool selected, ImVec2 size);
        static void tabPanel(const TabPanel& panel, uint32_t& activeTabIndex);
    private:
        void createDesriptorPool();
        void setupStyle(float alpha, float scale);

        Context*       mContext  = nullptr;
        DescriptorPool mDescriptorPool{};
        ImDrawData*    mDrawData = nullptr;
    };
}

#endif //COLLECTION_IMGUI_HPP