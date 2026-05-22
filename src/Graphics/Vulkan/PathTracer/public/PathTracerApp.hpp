//
// Created by igor on 4/22/26.
//

#ifndef COLLECTION_PATHTRACERAPP_HPP
#define COLLECTION_PATHTRACERAPP_HPP

#include "PathTracer.hpp"
#include "Rasterizer.hpp"
#include "Outliner.hpp"
#include "Context.hpp"
#include "Camera.hpp"
#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Semaphore.hpp"
#include "Fence.hpp"
#include "ImGui.hpp"

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
        void pixelClicked(uint32_t x, uint32_t y);
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
        void createImages();
        void createTextures();
        void loadScene();
        void loadModel(uint32_t modelIndex, const std::string& path);
        void createPathTracer();
        void createOutliner();
        void createImGui();
        void update();
        void recordRaster();
        void recordTracer();
        void recordOutliner(uint32_t imageIndex);
        void recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer);
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
        void updateInstanceModel(uint32_t instanceIndex, const Transform& transform);
        GBuffer& currentGBuffer() { return mGBuffers[mCurrentFrame]; }

#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        uint32_t mFramesInFlight = 3;
        uint32_t mCurrentFrame   = 0;
        uint32_t mFrameCount     = 0;
        scene::FlyCamera     mFlyCamera{};
        scene::OrbitalCamera mOrbitalCamera{};
        scene::AbsCamera*    mCamera = nullptr;
        AlignedDirectLight   mDirectLight{};
        bool mRenderImGui = false;
        int  mSPP         = 1;
        int  mMinDepth    = 0;
        int  mMaxDepth    = 1;
        int  mDisplayMode = 4;
        UIVec2 mPixel{UINT32_MAX, UINT32_MAX};

        json                   mScene{};
        Context                mContext{};
        CommandPool            mTracerCommandPool{};
        CommandBuffers         mTracerCommandBuffers{};
        CommandPool            mRasterCommandPool{};
        CommandBuffers         mRasterCommandBuffers{};
        CommandPool            mOutlinerCommandPool{};
        CommandBuffers         mOutlinerCommandBuffers{};
        Image                  mTracerImage{};
        ImageView              mTracerView{};
        Image                  mPresentImage{};
        ImageView              mPresentView{};
        Swapchain              mSwapchain{};
        std::vector<VkImage>   mSwapchainImages{};
        std::vector<ImageView> mSwapchainImageViews{};
        std::vector<Semaphore> mImageAvailableSemaphores{};
        std::vector<Semaphore> mRasterFinishedSemaphores{};
        std::vector<Semaphore> mTracerFinishedSemaphores{};
        std::vector<Semaphore> mOutlinerFinishedSemaphores{};
        std::vector<Fence>     mFences{};

        std::vector<Vertex>               mVertices{};
        std::vector<uint32_t>             mIndices{};
        std::vector<AlignedTriangle>      mTriangles{};
        std::vector<AlignedTriangleExtra> mTriangleExtras{};
        std::vector<AlignedNode>          mNodes{};
        std::vector<AlignedNode>          mTLASNodes{};
        std::vector<MeshData>             mMeshesData{};
        std::vector<MeshPrimitive>        mMeshPrimitives{};
        std::vector<MeshInstance>         mRasterInstances{};
        std::vector<MeshInstance>         mTracerInstances{};
        std::vector<cm::Material>         mMaterials{};
        std::vector<TexturesByType>       mTextures{};

        std::vector<GBuffer> mGBuffers{};
        Rasterizer mRasterizer{};
        PathTracer mPathTracer{};
        Outliner   mOutliner{};
        VkImGui    mImGui{};
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP