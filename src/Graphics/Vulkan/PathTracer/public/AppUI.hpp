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
#include "Model/RenderSettings.hpp"
#include <ImGuizmo.h>

namespace crv::graphics::vulkan {
    namespace cs = scene;
    struct AppUICreateInfo {
        Context*         context         = nullptr;
        Swapchain*       swapchain       = nullptr;
        ResourceManager* resourceManager = nullptr;
        RenderSettings*  renderSettings  = nullptr;
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

        void setUpdateImageCallBack(const std::function<void()>& callBack) { mUpdateImage = callBack; }
        void setCameraSetCallBack(const std::function<void(cs::CameraType type)>& callBack) { mCameraSet = callBack; }
        void setUploadTextureCallBack(const std::function<void(const std::string& path, uint32_t materialIndex, int textureType)>& callBack) { mUploadTexture = callBack; }
        void setLoadSkyboxCallBack(const std::function<void(const std::string& path)>& callBack) { mLoadSkybox = callBack; }
        void setRemoveSkyboxCallBack(const std::function<void()>& callBack) { mRemoveSkybox = callBack; }
        void setSaveImageCallBack(const std::function<void()>& callBack) { mSaveImage = callBack; }
        void setSaveSceneCallBack(const std::function<void()>& callBack) { mSaveScene = callBack; }
        void setDuplicateInstancesCallBack(const std::function<void(const std::vector<uint32_t>& indices)>& callBack) { mDuplicateInstances = callBack; }
        void setRemoveInstancesCallBack(const std::function<void(const std::vector<uint32_t>& indices)>& callBack) { mRemoveInstances = callBack; }
        void setRegionSelectCallBack(const std::function<void(int x0, int y0, int x1, int y1, bool additive)>& callBack) { mRegionSelect = callBack; }
        void setSelectInstanceCallBack(const std::function<void(uint32_t index, bool additive)>& callBack) { mSelectInstance = callBack; }
        void setAddMaterialCallBack(const std::function<void(uint32_t instanceIndex)>& callBack) { mAddMaterial = callBack; }
    private:
        void drawGizmo(const AppUIDrawInfo& info);
        void handleMarquee();
        void drawCursorDot();
        void drawOverView(const AppUIDrawInfo& info);
        void drawCameraTab(const AppUIDrawInfo& info);
        void drawRenderTab(const AppUIDrawInfo& info);
        void drawObjectTab(const AppUIDrawInfo& info);
        void drawSettings(const AppUIDrawInfo& info);

        Context*         mContext         = nullptr;
        Swapchain*       mSwapchain       = nullptr;
        ResourceManager* mResourceManager = nullptr;
        RenderSettings*  mSettings        = nullptr;
        VkImGui          mImGui           = CRV_NULL_HANDLE;

        ImGuizmo::OPERATION mGizmoOp      = ImGuizmo::TRANSLATE;
        ImGuiFileDialog  mFileDialog{};

        std::function<void()> mUpdateImage{};
        std::function<void(cs::CameraType type)> mCameraSet{};
        std::function<void(const std::string& path, uint32_t materialIndex, int textureType)> mUploadTexture{};
        int              mUploadTextureType = 0;
        std::function<void(const std::string& path)> mLoadSkybox{};
        std::function<void()> mRemoveSkybox{};
        ImGuiFileDialog  mSkyboxFileDialog{};
        std::function<void()> mSaveImage{};
        std::function<void()> mSaveScene{};
        std::function<void(const std::vector<uint32_t>& indices)> mDuplicateInstances{};
        std::function<void(const std::vector<uint32_t>& indices)> mRemoveInstances{};
        std::function<void(int x0, int y0, int x1, int y1, bool additive)> mRegionSelect{};
        std::function<void(uint32_t index, bool additive)> mSelectInstance{};
        std::function<void(uint32_t instanceIndex)> mAddMaterial{};
        bool    mMarqueeActive = false;
        ImVec2  mMarqueeStart{};
    };
}

#endif //COLLECTION_APPUI_HPP