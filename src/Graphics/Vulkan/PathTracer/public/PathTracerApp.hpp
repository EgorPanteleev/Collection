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
        [[nodiscard]] cs::AbsCamera* camera() const { return mCamera; }
        [[nodiscard]] Window& window() { return mContext.window(); }
    private:
        void updateCurrentFrame() { mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight; }
        void readScene(const std::string& scenePath);
        void createContext();
        void createSwapChain();
        void createImages();
        void createSwapChainImages();
        void createSyncObjects();
        void createCommandBuffers();
        void createCamera();
        void loadModel(uint32_t modelIndex, const std::string& path);
        void loadScene();
        void createRayTracerPass();
        void update();
        void recordTracer(uint32_t imageIndex);
        void recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer);
        void record(uint32_t imageIndex);
        void submit(uint32_t imageIndex);
        void acquireNextImage(uint32_t& imageIndex);
        void drawFrame();

        using VkASInstance = VkAccelerationStructureInstanceKHR;
#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        uint32_t                  mFramesInFlight = 3;
        uint32_t                  mCurrentFrame   = 0;

        json                      mScene{};
        Context                   mContext              = CRV_NULL_HANDLE;
        Swapchain                 mSwapchain            = CRV_NULL_HANDLE;
        RayTracerPass             mRayTracerPass        = CRV_NULL_HANDLE;

        std::vector<VkImage>      mSwapchainImages{};
        std::vector<ImageView>    mSwapchainImageViews{};
        Image                     mTracerImage          = CRV_NULL_HANDLE;
        ImageView                 mTracerView           = CRV_NULL_HANDLE;

        std::vector<Fence>        mFences{};
        std::vector<Semaphore>    mImageAvailableSemaphores{};
        std::vector<Semaphore>    mTracerFinishedSemaphores{};

        CommandPool               mTracerCommandPool    = CRV_NULL_HANDLE;
        CommandBuffers            mTracerCommandBuffers = CRV_NULL_HANDLE;

        cs::FlyCamera             mFlyCamera{};
        cs::OrbitalCamera         mOrbitalCamera{};
        cs::AbsCamera*            mCamera               = nullptr;

        std::vector<BLASEntry>    mBLASEntries{};
        std::vector<VkASInstance> mInstances{};
        Buffer                    mInstanceBuffer       = CRV_NULL_HANDLE;
        AccelerationStructure     mTLAS                 = CRV_NULL_HANDLE;
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP