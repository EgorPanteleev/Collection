//
// Created by igor on 4/22/26.
//

#ifndef COLLECTION_PATHTRACERAPP_HPP
#define COLLECTION_PATHTRACERAPP_HPP

#include "PathTracer.hpp"
#include "Context.hpp"
#include "Camera.hpp"
#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Semaphore.hpp"
#include "Fence.hpp"
#include "ImGui.hpp"

namespace crv::graphics::vulkan {
    struct PathTracerAppCreateInfo {
        WindowCreateInfo windowCreateInfo{};
        scene::CameraCreateInfo cameraCreateInfo{};
        glm::mat4 modelMatrix;
        std::string modelPath;
        AlignedDirectLight directLight{};
    };

    class PathTracerApp {
    public:
        PathTracerApp(const PathTracerAppCreateInfo& info);
        void run();
        void toggleControlPanel() { mRenderImGui = !mRenderImGui; }
        void updateImage() { mFrameCount = 0; }
        [[nodiscard]] Window& window() { return mContext.window(); }
        [[nodiscard]] scene::AbsCamera* camera() const { return mCamera; }
    protected:
        void createCamera(const scene::CameraCreateInfo& createInfo);
        void createContext(const WindowCreateInfo& windowCreateInfo);
        void createCommandPool();
        void createCommandBuffers();
        void createSwapChain();
        void createSwapChainImages();
        void createSyncObjects();
        void createPresentImage();
        void createPathTracer(const PathTracerAppCreateInfo& createInfo);
        void createImGui();
        void update();
        void record(uint32_t imageIndex);
        void submit(uint32_t imageIndex);
        void acquireNextImage(uint32_t& imageIndex);
        void drawFrame();
        void drawControlPanel();
        void updateCurrentFrame() { mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight; }
        void setCamera(scene::CameraType type);

#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        uint32_t mFramesInFlight = 2;
        uint32_t mCurrentFrame = 0;
        uint32_t mFrameCount = 0;
        scene::FlyCamera mFlyCamera{};
        scene::OrbitalCamera mOrbitalCamera{};
        scene::AbsCamera* mCamera = nullptr;
        AlignedDirectLight mDirectLight{};
        bool mRenderImGui = false;

        Context mContext{};
        CommandPool mCommandPool{};
        CommandBuffers mCommandBuffers{};
        Image mPresentImage{};
        ImageView mPresentImageView{};
        Swapchain mSwapchain{};
        std::vector<VkImage> mSwapchainImages{};
        std::vector<ImageView> mSwapchainImageViews{};
        std::vector<Semaphore> mImageAvailableSemaphores{};
        std::vector<Semaphore> mComputeFinishedSemaphores{};
        std::vector<Fence> mFences{};
        PathTracer mPathTracer{};
        VkImGui mImGui{};
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP