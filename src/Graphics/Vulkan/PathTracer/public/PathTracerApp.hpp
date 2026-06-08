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
#include "Semaphore.hpp"
#include "Fence.hpp"
#include "Types.hpp"
#include "TypesGPU.hpp"
#include "ImGui.hpp"

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
        void pixelClicked(uint32_t x, uint32_t y);
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
        void loadModel(uint32_t modelIndex, const std::string& path);
        void loadScene();
        void createTextures();
        void createRayTracerPass();
        void createImGui();
        void update();
        void recordTracer(uint32_t imageIndex);
        void recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer);
        void recordPixelRead(VkCommandBuffer commandBuffer);
        void updateSelectedInstance();
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
        uint32_t                     mSelectedInstanceId   = UINT32_MAX;
        ivec2                        mClickedPixel         = {UINT32_MAX, UINT32_MAX};

        json                         mScene{};
        Context                      mContext              = CRV_NULL_HANDLE;
        Swapchain                    mSwapchain            = CRV_NULL_HANDLE;
        RayTracerPass                mRayTracerPass        = CRV_NULL_HANDLE;

        std::vector<VkImage>         mSwapchainImages{};
        std::vector<ImageView>       mSwapchainImageViews{};
        Image                        mTracerImage          = CRV_NULL_HANDLE;
        ImageView                    mTracerView           = CRV_NULL_HANDLE;
        Image                        mInstanceIdImage      = CRV_NULL_HANDLE;
        ImageView                    mInstanceIdView       = CRV_NULL_HANDLE;
        Buffer                       mReadbackBuffer       = CRV_NULL_HANDLE;

        std::vector<Fence>           mFences{};
        std::vector<Semaphore>       mImageAvailableSemaphores{};
        std::vector<Semaphore>       mTracerFinishedSemaphores{};

        CommandPool                  mTracerCommandPool    = CRV_NULL_HANDLE;
        CommandBuffers               mTracerCommandBuffers = CRV_NULL_HANDLE;

        cs::FlyCamera                mFlyCamera{};
        cs::OrbitalCamera            mOrbitalCamera{};
        cs::AbsCamera*               mCamera               = nullptr;

        std::vector<BLASEntry>       mBLASEntries{};
        std::vector<BLASInfo>        mBLASInfos{};
        std::vector<VkASInstance>    mInstances{};
        std::vector<InstanceInfoGPU> mInstanceInfos{};
        Buffer                       mInstanceBuffer       = CRV_NULL_HANDLE;
        AccelerationStructure        mTLAS                 = CRV_NULL_HANDLE;
        std::vector<cm::Material>    mMaterials{};
        std::vector<TexturesByType>  mTextures{};
        DirectLightGPU               mDirectLight{};

        VkImGui                      mImGui                = CRV_NULL_HANDLE;
        int                          mSPP                  = 1;
        int                          mMinDepth             = 0;
        int                          mMaxDepth             = 1;
        int                          mDisplayMode          = 4;
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP