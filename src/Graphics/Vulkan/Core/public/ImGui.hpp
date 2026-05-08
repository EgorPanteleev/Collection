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
    };

    struct ImGuiRenderInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkImageView     imageView = VK_NULL_HANDLE;
        VkExtent2D      extent{};
    };

    class VkImGui {
    public:
        VkImGui() = default;
        explicit VkImGui(const ImGuiCreateInfo& info);
        VkImGui(VkImGui&&);
        ~VkImGui();
        VkImGui& operator=(VkImGui&&) noexcept;
        void destroy();
        void beginFrame();
        void endFrame();
        void render(const ImGuiRenderInfo& info);
        static void demo();
        static bool selectableButton(const char* label, bool cond);
        static void beginGroup(const char* name);
        static void endGroup();
        [[nodiscard]] ImDrawData* drawData() { return mDrawData; }
    private:
        void createDesriptorPool();
        void setupStyle(float alpha);

        Context*       mContext  = nullptr;
        DescriptorPool mDescriptorPool{};
        ImDrawData*    mDrawData = nullptr;
    };
}

#endif //COLLECTION_IMGUI_HPP