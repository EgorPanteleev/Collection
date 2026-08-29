//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_PATHTRACERAPP_HPP
#define COLLECTION_PATHTRACERAPP_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "Context.hpp"
#include "Camera.hpp"
#include "Swapchain.hpp"
#include "Image.hpp"
#include "ImageView.hpp"
#include "RayTracerPass.hpp"
#include "RasterizerPass.hpp"
#include "PostprocessPass.hpp"
#include "Semaphore.hpp"
#include "Fence.hpp"
#include "Types.hpp"
#include "ResourceManager.hpp"
#include "AppUI.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    struct PathTracerAppCreateInfo {
        std::string scenePath{};
    };

    class PathTracerApp {
    public:
        PathTracerApp() = delete;
        explicit PathTracerApp(const PathTracerAppCreateInfo& createInfo);
        void run();
        void updateImage() { mFrameCount = 0; }
        void onCameraMoved() { mFrameCount = 0; mCameraMoved = true; }
        void pixelClicked(uint32_t x, uint32_t y, bool additive);
        void clearSelection();
        void regionSelect(int x0, int y0, int x1, int y1, bool additive);
        void toggleControlPanel() { mRenderImGui = !mRenderImGui; }
        [[nodiscard]] cs::AbsCamera* camera() const { return mCamera; }
        [[nodiscard]] Window& window() { return mContext.window(); }
    private:
        void updateCurrentFrame() { mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight; }
        void readScene(const std::string& scenePath);
        void createContext();
        void createSwapChain();
        void createBuffers();
        void createImages();
        void createSwapChainImages();
        void createSyncObjects();
        void createCommandBuffers();
        void createCamera();
        void createResourceManager();
        void loadScene();
        void createRayTracerPass();
        void createRasterizerPass();
        void createPostprocessPass();
        void createUI();
        void update();
        void recordTracer();
        void recordRaster();
        void recordPostprocess();
        void recordPresent(uint32_t imageIndex);
        void recordPixelRead(VkCommandBuffer commandBuffer);
        void saveImage();
        void saveScene();
        void updateSelectedInstance();
        void applySelection(uint32_t id, bool additive);
        void record(uint32_t imageIndex);
        void submit(uint32_t imageIndex);
        void acquireNextImage(uint32_t& imageIndex);
        void setCamera(scene::CameraType type);
        void drawControlPanel();
        void drawFrame();

        using VkASInstance = VkAccelerationStructureInstanceKHR;
#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        bool                         mRenderImGui          = false;
        uint32_t                     mFramesInFlight       = 1;
        uint32_t                     mCurrentFrame         = 0;
        uint32_t                     mFrameCount           = 0;
        bool                         mCameraMoved          = false;
        uint32_t                     mEffectiveScale       = 1;
        std::vector<uint32_t>        mSelectedInstances{};
        uint32_t                     mActiveInstance       = UINT32_MAX;
        bool                         mAdditiveSelect       = false;
        bool                         mPendingSelection     = false;
        ivec2                        mClickedPixel         = {UINT32_MAX, UINT32_MAX};

        json                         mJson{};
        Context                      mContext              = CRV_NULL_HANDLE;
        Swapchain                    mSwapchain            = CRV_NULL_HANDLE;
        RayTracerPass                mRayTracerPass        = CRV_NULL_HANDLE;
        RasterizerPass               mRasterizerPass       = CRV_NULL_HANDLE;
        PostprocessPass              mPostprocessPass      = CRV_NULL_HANDLE;

        std::vector<VkImage>         mSwapchainImages{};
        std::vector<ImageView>       mSwapchainImageViews{};
        Image                        mTracerImage          = CRV_NULL_HANDLE;
        ImageView                    mTracerView           = CRV_NULL_HANDLE;
        Image                        mTracerInstanceImage  = CRV_NULL_HANDLE;
        ImageView                    mTracerInstanceView   = CRV_NULL_HANDLE;
        Image                        mRasterInstanceImage  = CRV_NULL_HANDLE;
        ImageView                    mRasterInstanceView   = CRV_NULL_HANDLE;
        Image                        mFinalImage           = CRV_NULL_HANDLE;
        ImageView                    mFinalView            = CRV_NULL_HANDLE;
        Buffer                       mReadbackBuffer       = CRV_NULL_HANDLE;

        std::vector<Fence>           mFences{};
        std::vector<Semaphore>       mImageAvailableSemaphores{};
        std::vector<Semaphore>       mTracerFinishedSemaphores{};
        std::vector<Semaphore>       mRasterFinishedSemaphores{};
        std::vector<Semaphore>       mPostprocessFinishedSemaphores{};

        CommandPool                  mTracerCommandPool    = CRV_NULL_HANDLE;
        CommandBuffers               mTracerCommandBuffers = CRV_NULL_HANDLE;
        CommandPool                  mRasterCommandPool    = CRV_NULL_HANDLE;
        CommandBuffers               mRasterCommandBuffers = CRV_NULL_HANDLE;
        CommandPool                  mPostprocessCommandPool    = CRV_NULL_HANDLE;
        CommandBuffers               mPostprocessCommandBuffers = CRV_NULL_HANDLE;

        cs::FlyCamera                mFlyCamera{};
        cs::OrbitalCamera            mOrbitalCamera{};
        cs::AbsCamera*               mCamera               = nullptr;

        ResourceManager              mResourceManager{};

        RenderSettings               mRenderSettings{};
        AppUI                        mUI{};
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP