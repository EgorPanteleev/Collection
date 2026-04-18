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
//#include "BVH.hpp"

namespace crv::graphics::vulkan {
    using Scalar = float;
    using Vec4 = glm::vec<4, Scalar>;
    struct AlignedTriangle {
        Vec4 p0, e1, e2, N;
    };

    struct AlignedCamera {
        Vec4 position, forward, right, up;
        float FOV, aspectRatio, nearPlane, farPlane;
    };

    struct PushConstants {
        int width;
        int height;
        int triangleCount;
    };

    struct PathTracerCreateInfo {
        const WindowCreateInfo& windowCreateInfo;
        const scene::CameraCreateInfo cameraCreateInfo;
        std::vector<AlignedTriangle> triangles;
        //BVH<>
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
        std::vector<AlignedTriangle> mTriangles{};
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
        Buffer mCameraBuffer{};
        Buffer mTrianglesBuffer{};
        Image mdImage{};
        ImageView mdImageView{};
        Swapchain mSwapchain{};
        std::vector<VkImage> mImages{};
        std::vector<ImageView> mImageViews{};
        Semaphore mImageAvailableSemaphore{};
        Semaphore mComputeFinishedSemaphore{};
        Fence mFence{};
    };
}

#endif //COLLECTION_PATHTRACER_HPP