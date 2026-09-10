//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_RENDERER_HPP
#define COLLECTION_RENDERER_HPP

#include "Context.hpp"
#include "Swapchain.hpp"
#include "Image.hpp"
#include "ImageView.hpp"
#include "RayTracerPass.hpp"
#include "RasterizerPass.hpp"
#include "PostprocessPass.hpp"
#include "Semaphore.hpp"
#include "Fence.hpp"
#include "Types.hpp"
#include "Model/Model.hpp"
#include "ResourceManager.hpp"
#include "CommandStream.hpp"
#include "View/AppUI.hpp"

#include <glm/glm.hpp>

namespace crv::graphics::vulkan {
    struct RendererCreateInfo {
        WindowCreateInfo windowCreateInfo{};
        Model*           model    = nullptr;
        CommandStream*   commands = nullptr;
    };

    class Renderer {
    public:
        Renderer() = default;
        explicit Renderer(const RendererCreateInfo& info);
        Renderer(const Renderer&) = delete;
        Renderer& operator=(const Renderer&) = delete;
        Renderer(Renderer&&) = delete;
        Renderer& operator=(Renderer&&) = delete;
        [[nodiscard]] Window& window() { return mContext.window(); }
        [[nodiscard]] auto    extent() { return mSwapchain.extent(); }
        void initUI();
        void beginFrame();
        void endFrame();
        void onCameraMoved() { mFrameCount = 0; mCameraMoved = true; }
        void waitIdle() { vkDeviceWaitIdle(mContext.device()); }
        void pick(glm::dvec2 cursor, bool additive);
        void saveImage();
        void toggleUI() { mRenderImGui = !mRenderImGui; }
    private:
        void updateCurrentFrame() { mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight; }
        void createContext(const WindowCreateInfo& window);
        void createSwapChain();
        void createBuffers();
        void createImages();
        void createSwapChainImages();
        void createSyncObjects();
        void createCommandBuffers();
        void createResourceManager();
        void createRayTracerPass();
        void createRasterizerPass();
        void createPostprocessPass();
        void update();
        void recordTracer();
        void recordRaster();
        void recordPostprocess();
        void recordPresent(uint32_t imageIndex);
        void recordPixelRead(VkCommandBuffer commandBuffer);
        void updateSelectedInstance();
        void drawControlPanel();
        void flushUpdates();
        void record(uint32_t imageIndex);
        void submit(uint32_t imageIndex);
        void acquireNextImage(uint32_t& imageIndex);

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
        bool                         mAdditiveSelect       = false;
        ivec2                        mClickedPixel         = {UINT32_MAX, UINT32_MAX};

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

        Model*                       mModel                = nullptr;
        CommandStream*               mCommands             = nullptr;
        ResourceManager              mResourceManager{};
        AppUI                        mUI{};
    };
}

#endif //COLLECTION_RENDERER_HPP
