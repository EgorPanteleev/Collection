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
#include "Types.hpp"
#include "ResourceManager.hpp"
#include <ImGuizmo.h>

namespace crv::graphics::vulkan {
    namespace cs = scene;
    struct AppUICreateInfo {
        Context*         context         = nullptr;
        Swapchain*       swapchain       = nullptr;
        ResourceManager* resourceManager = nullptr;
    };

    struct AppUIRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        ImageView*      imageView     = nullptr;
    };

    struct AppUIDrawInfo {
        bool           drawUI                = false;
        cs::AbsCamera* camera                = nullptr;
        uint32_t       selectedInstanceIndex = 0;
        uint32_t       frameCount            = 0;
    };

    class AppUI {
    public:
        AppUI() = default;
        explicit AppUI(const AppUICreateInfo& info);
        void record(const AppUIRecordInfo& info);
        void draw(const AppUIDrawInfo& info);

        void setUpdateImageCallBack(const std::function<void()>& callBack) { mUpdateImage = callBack; }
        void setCameraSetCallBack(const std::function<void(cs::CameraType type)>& callBack) { mCameraSet = callBack; }
        void setUploadTextureCallBack(const std::function<void(const std::string& path, uint32_t materialIndex)>& callBack) { mUploadTexture = callBack; }

        [[nodiscard]] uint32_t spp() const { return mSPP; }
        [[nodiscard]] uint32_t minDepth() const { return mMinDepth; }
        [[nodiscard]] uint32_t maxDepth() const { return mMaxDepth; }
        [[nodiscard]] uint32_t displayMode() const { return mDisplayMode; }
        [[nodiscard]] bool     nee() const { return mNee; }
    private:
        void drawGizmo(const AppUIDrawInfo& info);
        void drawCursorDot();
        void drawOverView(const AppUIDrawInfo& info);
        void drawCameraTab(const AppUIDrawInfo& info);
        void drawRenderTab(const AppUIDrawInfo& info);
        void drawObjectTab(const AppUIDrawInfo& info);
        void drawSettings(const AppUIDrawInfo& info);

        Context*         mContext         = nullptr;
        Swapchain*       mSwapchain       = nullptr;
        ResourceManager* mResourceManager = nullptr;
        VkImGui          mImGui           = CRV_NULL_HANDLE;

        int              mSPP             = 1;
        int              mMinDepth        = 0;
        int              mMaxDepth        = 1;
        int              mDisplayMode     = 4;
        bool             mNee             = true;

        ImGuizmo::OPERATION mGizmoOp      = ImGuizmo::TRANSLATE;
        ImGuiFileDialog  mFileDialog{};

        std::function<void()> mUpdateImage{};
        std::function<void(cs::CameraType type)> mCameraSet{};
        std::function<void(const std::string& path, uint32_t materialIndex)> mUploadTexture{};
    };
}

#endif //COLLECTION_APPUI_HPP