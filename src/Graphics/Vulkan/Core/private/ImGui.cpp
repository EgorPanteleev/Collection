//
// Created by igor on 4/23/26.
//

#include "ImGui.hpp"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "IconsFontAwesome6.h"
#include "fa_solid_900.h"

namespace crv::graphics::vulkan {
    static ImFontConfig makeDefaultFontConfig() {
        ImFontConfig config{};
        config.FontDataOwnedByAtlas = true;
        config.OversampleH = 0;
        config.OversampleV = 0;
        config.ExtraSizeScale = 1.0f;
        config.GlyphMaxAdvanceX = FLT_MAX;
        config.RasterizerMultiply = 1.0f;
        config.RasterizerDensity = 1.0f;
        config.EllipsisChar = 0;
        return config;
    }

    VkImGui::VkImGui(const ImGuiCreateInfo &info) : mContext(info.context) {
        createDesriptorPool();
        //context setup
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void) io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.MouseDrawCursor = true;
        setupStyle(info.alpha, info.scale);
        //init glfw backend
        bool installCallbacks = true;
        ImGui_ImplGlfw_InitForVulkan(mContext->glfwWindow(), installCallbacks);

        //init vulkan backend
        ImGui_ImplVulkan_InitInfo init_info = {
            .Instance = mContext->instance(),
            .PhysicalDevice = mContext->physicalDevice(),
            .Device = mContext->device(),
            .QueueFamily = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .Queue = mContext->queue(QueueFamilyType::GRAPHICS),
            .DescriptorPool = mDescriptorPool.get(),
            .MinImageCount = info.imageCount,
            .ImageCount = info.imageCount,
            .PipelineCache = VK_NULL_HANDLE,
            .PipelineInfoMain = {
                .RenderPass = VK_NULL_HANDLE,
                .PipelineRenderingCreateInfo = {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                    .colorAttachmentCount = 1,
                    .pColorAttachmentFormats = &info.format,
                    .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
                    .stencilAttachmentFormat = VK_FORMAT_UNDEFINED
                }
            },
            .UseDynamicRendering = true,
        };
        ImGui_ImplVulkan_Init(&init_info);
        //ImGui_ImplVulkan_CreateFontsTexture();
    }

    VkImGui::VkImGui(VkImGui&& other) {
        mContext = other.mContext;
        other.mContext = nullptr;
        mDescriptorPool = std::move(other.mDescriptorPool);
        mDrawData = other.mDrawData;
        mDrawData = nullptr;
    }

    VkImGui& VkImGui::operator=(VkImGui&& other) noexcept {
        if (this != &other) {
            destroy();
            mContext = other.mContext;
            other.mContext = nullptr;
            mDescriptorPool = std::move(other.mDescriptorPool);
            mDrawData = other.mDrawData;
            mDrawData = nullptr;
        }
        return *this;
    }

    VkImGui::~VkImGui() {
        destroy();
    }

    void VkImGui::destroy() {
        if (mContext == nullptr) return;
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void VkImGui::createDesriptorPool() {
        constexpr uint32_t size = 10;
        const std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_SAMPLER               , size},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, size},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE         , size},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE         , size},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER  , size},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER  , size},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER        , size},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , size},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, size},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, size},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT      , size}
        };
        const DescriptorPoolCreateInfo createInfo{
            .device = mContext->device(),
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .poolSizes = poolSizes,
            .maxSets = static_cast<uint32_t>(size * poolSizes.size())
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void VkImGui::setupStyle(const float alpha, const float scale) {
        ImGuiStyle& style = ImGui::GetStyle();
        ImVec4* colors = style.Colors;
        ImGuiIO& io = ImGui::GetIO();

        colors[ImGuiCol_Text]                  = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
        colors[ImGuiCol_TextDisabled]          = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
        colors[ImGuiCol_WindowBg]              = ImVec4(0.08f, 0.08f, 0.08f, alpha);
        colors[ImGuiCol_ChildBg]               = ImVec4(0.10f, 0.10f, 0.10f, alpha);
        colors[ImGuiCol_PopupBg]               = ImVec4(0.11f, 0.11f, 0.11f, 0.94f);
        colors[ImGuiCol_Border]                = ImVec4(0.25f, 0.25f, 0.25f, 0.50f);
        colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]               = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
        colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        colors[ImGuiCol_FrameBgActive]         = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
        colors[ImGuiCol_TitleBg]               = ImVec4(0.07f, 0.07f, 0.07f, 1.00f);
        colors[ImGuiCol_TitleBgActive]         = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.05f, 0.05f, 0.05f, 0.75f);
        colors[ImGuiCol_MenuBarBg]             = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
        colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);
        colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.35f, 0.35f, 0.35f, 0.90f);
        colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
        colors[ImGuiCol_CheckMark]             = ImVec4(0.56f, 0.90f, 0.69f, 1.00f);
        colors[ImGuiCol_SliderGrab]            = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.55f, 0.55f, 0.55f, 1.00f);
        colors[ImGuiCol_Button]                = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
        colors[ImGuiCol_ButtonHovered]         = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        colors[ImGuiCol_ButtonActive]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
        colors[ImGuiCol_Header]                = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        colors[ImGuiCol_HeaderHovered]         = ImVec4(0.25f, 0.25f, 0.25f, 0.80f);
        colors[ImGuiCol_HeaderActive]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
        colors[ImGuiCol_Separator]             = ImVec4(0.35f, 0.35f, 0.35f, 0.50f);
        colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.55f, 0.55f, 0.55f, 0.78f);
        colors[ImGuiCol_SeparatorActive]       = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        colors[ImGuiCol_ResizeGrip]            = ImVec4(0.30f, 0.30f, 0.30f, 0.30f);
        colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.35f, 0.35f, 0.35f, 0.60f);
        colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.40f, 0.40f, 0.40f, 0.90f);
        colors[ImGuiCol_Tab]                   = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
        colors[ImGuiCol_TabHovered]            = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        colors[ImGuiCol_TabActive]             = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
        colors[ImGuiCol_TabUnfocused]          = ImVec4(0.07f, 0.07f, 0.07f, 1.00f);
        colors[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.24f, 0.54f, 0.33f, 0.45f);
        colors[ImGuiCol_DragDropTarget]        = ImVec4(0.90f, 0.30f, 0.30f, 0.90f);
        colors[ImGuiCol_NavHighlight]          = ImVec4(0.42f, 0.75f, 0.58f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.15f, 0.15f, 0.15f, 0.40f);

        style.WindowRounding    = 6.0f;
        style.FrameRounding     = 4.0f;
        style.ScrollbarRounding = 3.0f;
        style.GrabRounding      = 3.0f;
        style.TabRounding       = 4.0f;

        ImFontConfig textConfig = makeDefaultFontConfig();
        textConfig.FontDataOwnedByAtlas = false;
        io.Fonts->AddFontFromFileTTF(
            PROJECT_PATH"assets/fonts/JetBrainsMono-Regular.ttf",
            15.0f * scale,
            &textConfig,
            nullptr
        );

        ImFontConfig iconConfig = makeDefaultFontConfig();
        iconConfig.FontDataOwnedByAtlas = false;
        iconConfig.MergeMode = true;
        iconConfig.PixelSnapH = true;
        static const ImWchar iconRanges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
        io.Fonts->AddFontFromMemoryTTF(
        (void*)fa_solid_900,
        fa_solid_900_len,
        11.0f * scale,
        &iconConfig,
        iconRanges
        );
    }

    bool VkImGui::beginFrame() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        return true;
    }

    void VkImGui::endFrame() {
        ImGui::Render();
        mDrawData = ImGui::GetDrawData();
    }

    bool VkImGui::selectableButton(const char* label, bool cond) {
        if (cond) {
            ImVec4 baseColor = {0.4f, 0.4f, 0.4f, 1.0f};
            ImGui::PushStyleColor(ImGuiCol_Button,        baseColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(baseColor.x + 0.1f, baseColor.y + 0.1f, baseColor.z + 0.1f, baseColor.w));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(baseColor.x - 0.1f, baseColor.y - 0.1f, baseColor.z - 0.1f, baseColor.w));
        }
        bool ret = ImGui::Button(label) && !cond;
        if (cond) ImGui::PopStyleColor(3);

        return ret;
    }

    bool VkImGui::beginGroup(const char* name) {
        ImGui::Dummy({0.0f, 2.0f});
        separatorText(name);
        ImGui::Indent(4.0f);
        return true;
    }

    void VkImGui::endGroup() {
        ImGui::Unindent(4.0f);
    }

    void VkImGui::separatorText(const char* label) {
        ImGui::PushStyleVar(ImGuiStyleVar_SeparatorTextBorderSize, 0.5f);
        ImGui::SeparatorText(label);
        ImGui::PopStyleVar();
    }

    static bool inspectorBeginKeyValueTableWithCellPadding(const char* tableId, ImVec2 cellPadding) {
        if (!tableId) return false;

        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, cellPadding);
        ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings;
        if (!ImGui::BeginTable(tableId, 2, flags, ImVec2(-1.0f, 0.0f), 0.0f)) {
            ImGui::PopStyleVar();
            return false;
        }

        const ImGuiStyle* style = &ImGui::GetStyle();
        float labelWidth = ImGui::CalcTextSize("Accumulation").x;
        if (style) {
            labelWidth += style->CellPadding.x * 2.0f;
        }
        if (labelWidth < 86.0f) labelWidth = 86.0f;

        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoHide, labelWidth, 0);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoHide, 0.0f, 0);
        return true;
    }

    bool VkImGui::beginCompactTable(const char* tableId, const float padding) {
        if (!tableId) return false;

        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        framePadding.y = 2.0f;
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);
        if (!inspectorBeginKeyValueTableWithCellPadding(tableId, {padding, 0.0f})) {
            ImGui::PopStyleVar();
            return false;
        }
        return true;
    }

    void VkImGui::endCompactTable() {
        ImGui::EndTable();
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }

    void VkImGui::row(const char* label, const char* value) {
        if (!label || !value) return;
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", label);
        ImGui::TableSetColumnIndex(1);
        ImGui::AlignTextToFramePadding();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(value);
        ImGui::PopTextWrapPos();
    }

    bool VkImGui::tabButton(const char* label, bool selected, ImVec2 size) {
        if (!label) return false;
        const ImVec4* baseColor = &ImGui::GetStyleColorVec4((int)selected ? ImGuiCol_TabActive : ImGuiCol_Tab);
        const ImVec4* hoveredColor = &ImGui::GetStyleColorVec4(ImGuiCol_TabHovered);
        const ImVec4* activeColor = &ImGui::GetStyleColorVec4((int)selected ? ImGuiCol_TabActive : ImGuiCol_TabHovered);

        ImGui::PushStyleColor(ImGuiCol_Button, *baseColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, *hoveredColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, *activeColor);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, (ImVec2){10.0f, 5.0f});
        ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, (ImVec2){0.5f, 0.5f});
        bool pressed = ImGui::Button(label, size);
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);
        return pressed;
    }

    void VkImGui::tabPanel(const TabPanel &panel, uint32_t& activeTabIndex) {
        const ImGuiStyle* style = &ImGui::GetStyle();
        const float spacing = style->ItemSpacing.x;
        const float availableWidth = ImGui::GetContentRegionAvail().x;
        const uint32_t tabCount = panel.size();
        float buttonWidth = (availableWidth - spacing * static_cast<float>(tabCount - 1u)) / static_cast<float>(tabCount);
        if (buttonWidth < 1.0f) buttonWidth = 1.0f;
        const std::function<void()>* exFunc = nullptr;
        for (uint32_t i = 0; i < tabCount; i++) {
            if (i > 0) ImGui::SameLine();
            const auto& [label, index, func] = panel[i];
            const bool selected = index == activeTabIndex;
            if (tabButton(label, selected, (ImVec2){buttonWidth, 0.0f})) {
                activeTabIndex = index;
            }
            if (selected) exFunc = &func;
        }
        if (exFunc) (*exFunc)();
        ImGui::Dummy((ImVec2){0.0f, 1.0f});
    }

    void VkImGui::render(const ImGuiRenderInfo& info) {
         VkClearValue clearColor{
            .color = { {0.0f, 0.0f, 0.0f, 1.0f} }
         };

         VkRenderingAttachmentInfoKHR colorAttachment{};
         colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
         colorAttachment.imageView = info.imageView;
         colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
         colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
         colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
         colorAttachment.clearValue = clearColor;

         VkRenderingInfoKHR renderingInfo{};
         renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
         renderingInfo.renderArea.offset = {0, 0};
         renderingInfo.renderArea.extent = info.extent;
         renderingInfo.layerCount = 1;
         renderingInfo.colorAttachmentCount = 1;
         renderingInfo.pColorAttachments = &colorAttachment;

         vkCmdBeginRendering(info.commandBuffer, &renderingInfo);

         ImGui_ImplVulkan_RenderDrawData(mDrawData, info.commandBuffer);

         vkCmdEndRendering(info.commandBuffer);
    }

    void VkImGui::loadConfigFile(const char *path) {
        ImGui::LoadIniSettingsFromDisk(path);
    }

    void VkImGui::saveConfigFile(const char *path) {
        ImGui::SaveIniSettingsToDisk(path);
    }
}
