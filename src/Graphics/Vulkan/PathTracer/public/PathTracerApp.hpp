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
#include "Rasterizer.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace crv::graphics::vulkan {
    struct PathTracerAppCreateInfo {
        std::string scenePath;
        AlignedDirectLight directLight{};
    };

    class PathTracerApp {
    public:
        PathTracerApp(const PathTracerAppCreateInfo& info);
        void run();
        void toggleControlPanel() { mRenderImGui = !mRenderImGui; }
        void updateImage();
        [[nodiscard]] Window& window() { return mContext.window(); }
        [[nodiscard]] scene::AbsCamera* camera() const { return mCamera; }
    protected:
        void createCamera();
        void createContext();
        void createCommandPool();
        void createCommandBuffers();
        void createSwapChain();
        void createSwapChainImages();
        void createSyncObjects();
        void createPresentImage();
        void createTextures();
        void loadModel(const PathTracerAppCreateInfo& info);
        void createPathTracer();
        void createImGui();
        void update();
        void recordRaster();
        void recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer);
        void recordTracer(uint32_t imageIndex);
        void record(uint32_t imageIndex);
        void submit(uint32_t imageIndex);
        void acquireNextImage(uint32_t& imageIndex);
        void drawFrame();
        void drawControlPanel();
        void updateCurrentFrame() { mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight; }
        [[nodiscard]] uint32_t previousFrame() const { return (mCurrentFrame + mFramesInFlight - 1) % mFramesInFlight; }
        void setCamera(scene::CameraType type);
        void createGBuffers();
        void createRasterizer();
        GBuffer& currentGBuffer() { return mGBuffers[mGBufferFrame]; }

#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        uint32_t mFramesInFlight = 2;
        uint32_t mCurrentFrame   = 0;
        uint32_t mGBufferFrame   = 0;
        uint32_t mFrameCount     = 0;
        scene::FlyCamera     mFlyCamera{};
        scene::OrbitalCamera mOrbitalCamera{};
        scene::AbsCamera*    mCamera = nullptr;
        AlignedDirectLight   mDirectLight{};
        bool mRenderImGui = false;
        int  mSPP         = 1;
        int  mMinDepth    = 0;
        int  mMaxDepth    = 1;

        json                   mScene{};
        Context                mContext{};
        CommandPool            mComputeCommandPool{};
        CommandBuffers         mComputeCommandBuffers{};
        CommandPool            mGraphicsCommandPool{};
        CommandBuffers         mGraphicsCommandBuffers{};
        Image                  mPresentImage{};
        ImageView              mPresentImageView{};
        Swapchain              mSwapchain{};
        std::vector<VkImage>   mSwapchainImages{};
        std::vector<ImageView> mSwapchainImageViews{};
        std::vector<Semaphore> mImageAvailableSemaphores{};
        std::vector<Semaphore> mRasterFinishedSemaphores{};
        std::vector<Semaphore> mTracerFinishedSemaphores{};
        std::vector<Fence>     mFences{};

        std::vector<AlignedTriangle>      mTriangles{};
        std::vector<AlignedTriangleExtra> mTriangleExtras{};
        std::vector<AlignedNode>          mNodes{};
        std::vector<AlignedNode>          mTLASNodes{};
        std::vector<MeshData>             mMeshesData{};
        std::vector<MeshInstance>         mRasterInstances{};
        std::vector<MeshInstance>         mTracerInstances{};
        std::vector<cm::Material>         mMaterials{};
        std::vector<TexturesByType>       mTextures{};

        std::vector<GBuffer> mGBuffers{};
        Rasterizer mRasterizer{};
        PathTracer mPathTracer{};
        VkImGui    mImGui{};
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP