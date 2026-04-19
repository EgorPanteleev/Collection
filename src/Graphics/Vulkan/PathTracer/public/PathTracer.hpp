//
// Created by igor on 4/8/26.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Context.hpp"
#include "DescriptorSetLayout.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSets.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "ComputePipelines.hpp"
#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Buffer.hpp"
#include "Swapchain.hpp"
#include "Image.hpp"
#include "ImageView.hpp"
#include "Semaphore.hpp"
#include "Fence.hpp"
#include "Triangle.hpp"
#include "Camera.hpp"
#include "GPUTypes.hpp"
//#include "BVH.hpp"

namespace crv::graphics::vulkan {
    struct PathTracerCreateInfo {
        WindowCreateInfo windowCreateInfo{};
        scene::CameraCreateInfo cameraCreateInfo{};
        std::vector<AlignedTriangle> triangles{};
        std::vector<AlignedNode> nodes{};
        uint32_t framesInFlight = 2;
    };

    class PathTracer {
    public:
        explicit PathTracer(const PathTracerCreateInfo& info);
        void run();
        Window& window() { return mContext.window(); }
        scene::AbsCamera* camera() { return mCamera.get(); }
    protected:
        void createContext(const WindowCreateInfo& windowCreateInfo);
        void createDescriptorSetLayout();
        void createDescriptorPool();
        void createDescriptorSets();
        void createPipelineLayout();
        void createShaders();
        void createComputePipelines();
        void createCommandPool();
        void createCommandBuffers();
        void createSwapChain();
        void createImages();
        void createSyncObjects();
        void createBuffers(const PathTracerCreateInfo& info);
        void updateCurrentFrame() { mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight; }
        void update();
        void record(uint32_t imageIndex);
        void submit(uint32_t imageIndex);

        [[nodiscard]] std::vector<VkDescriptorSetLayout> getDescriptorLayouts() const;
        [[nodiscard]] std::vector<VkPipelineLayout> getPipelineLayouts() const;
#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        uint32_t mFramesInFlight = 2;
        uint32_t mCurrentFrame = 0;
        std::unique_ptr<scene::AbsCamera> mCamera{};

        Context mContext{};
        DescriptorSetLayout mDescriptorSetLayout{};
        DescriptorPool mDescriptorPool{};
        DescriptorSets mDescriptorSets{};
        PipelineLayout mPipelineLayout{};
        ShaderModule mShader{};
        ComputePipelines mComputePipelines{};
        CommandPool mCommandPool{};
        CommandBuffers mCommandBuffers{};
        std::vector<Buffer> mCameraBuffers{};
        Buffer mTriangleBuffer{};
        Buffer mNodeBuffer{};
        Image mdImage{};
        ImageView mdImageView{};
        Swapchain mSwapchain{};
        std::vector<VkImage> mImages{};
        std::vector<ImageView> mImageViews{};
        std::vector<Semaphore> mImageAvailableSemaphores{};
        std::vector<Semaphore> mComputeFinishedSemaphores{};
        std::vector<Fence> mFences{};
    };
}

#endif //COLLECTION_PATHTRACER_HPP