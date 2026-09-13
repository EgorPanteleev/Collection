//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_APPUI_HPP
#define COLLECTION_APPUI_HPP

#include "ImGui.hpp"
#include "ImGuiFileDialog.hpp"
#include "Swapchain.hpp"
#include "ImageView.hpp"
#include "AbsCamera.hpp"
#include "View/Types.hpp"
#include "Model/Scene.hpp"
#include "Model/RenderSettings.hpp"
#include "Command.hpp"
#include "CommandStream.hpp"
#include <ImGuizmo.h>

namespace crv::graphics::vulkan {
    namespace cs = scene;
    struct AppUICreateInfo {
        Context*         context         = nullptr;
        Swapchain*       swapchain       = nullptr;
        RenderSettings*  renderSettings  = nullptr;
        CommandStream*   commands        = nullptr;
        const Scene*     scene           = nullptr;
    };

    struct AppUIRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        ImageView*      imageView     = nullptr;
    };

    struct AppUIDrawInfo {
        bool                         drawUI            = false;
        cs::AbsCamera*               camera            = nullptr;
        const std::vector<uint32_t>* selectedInstances = nullptr;
        uint32_t                     activeInstance    = UINT32_MAX;
        uint32_t                     frameCount        = 0;
        uint32_t                     renderScale       = 1;
    };

    class AppUI {
    public:
        AppUI() = default;
        explicit AppUI(const AppUICreateInfo& info);
        void record(const AppUIRecordInfo& info);
        void draw(const AppUIDrawInfo& info);

    private:
        void push(CommandType type, CommandPayload payload = EmptyPayload{}) {
            mCommands->push(Command{type, std::move(payload)});
        }
        void drawGizmo(const AppUIDrawInfo& info);
        void handleMarquee();
        void drawCursorDot();
        void drawOverView(const AppUIDrawInfo& info);
        void drawScene(const AppUIDrawInfo& info);
        void drawInstanceNode(uint32_t index, const std::vector<std::vector<uint32_t>>& children,
                              const AppUIDrawInfo& info);
        void drawCameraTab(const AppUIDrawInfo& info);
        void drawRenderTab(const AppUIDrawInfo& info);
        void drawObjectTab(const AppUIDrawInfo& info);
        void drawSettings(const AppUIDrawInfo& info);

        Context*         mContext         = nullptr;
        Swapchain*       mSwapchain       = nullptr;
        RenderSettings*  mSettings        = nullptr;
        CommandStream*   mCommands        = nullptr;
        const Scene*     mScene           = nullptr;
        VkImGui          mImGui           = CRV_NULL_HANDLE;

        ImGuizmo::OPERATION mGizmoOp      = ImGuizmo::TRANSLATE;
        ImGuiFileDialog  mFileDialog{};
        ImGuiFileDialog  mSkyboxFileDialog{};
        ImGuiFileDialog  mModelFileDialog{};
        int              mUploadTextureType = 0;
        bool             mUpdateImage = false;
        bool    mMarqueeActive = false;
        ImVec2  mMarqueeStart{};
    };
}

#endif //COLLECTION_APPUI_HPP