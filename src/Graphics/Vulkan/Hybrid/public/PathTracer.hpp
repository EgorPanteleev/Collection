//
// Created by igor on 4/8/26.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Context.hpp"
#include "DescriptorManager.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "ComputePipelines.hpp"
#include "Buffer.hpp"
#include "TypesGPU.hpp"

namespace crv::scene {
    class AbsCamera;
}

namespace crv::graphics::vulkan {
    struct PathTracerCreateInfo {
        Context* context;
        std::vector<TriangleGPU> triangles{};
        std::vector<TriangleExtraGPU> triangleExtras{};
        std::vector<NodeGPU> nodes{};
        std::vector<NodeGPU> tlasNodes{};
        std::vector<TexturesByType>* textures = nullptr;
        std::vector<MeshInstance> instances{};
        VkImage            outImage       = VK_NULL_HANDLE;
        VkImageView        outImageView   = VK_NULL_HANDLE;
        std::vector<GBuffer>* gBuffers{};
        uint32_t framesInFlight = 2;
    };

    struct PathTracerUpdateInfo {
        scene::AbsCamera*  camera         = nullptr;
        DirectLightGPU     directLight{};
        uint32_t           currentFrame   = 0;
    };

    struct PathTracerRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkExtent2D      extent{};
        uint32_t        currentFrame     = 0;
        PushConstants   constants{};
    };

    class PathTracer {
    public:
        PathTracer() = default;
        explicit PathTracer(const PathTracerCreateInfo& info);
        void update(const PathTracerUpdateInfo& info);
        void record(const PathTracerRecordInfo& info);
        void updateTLAS(const std::vector<NodeGPU>& nodes, const std::vector<MeshInstance>& instances, const std::vector<GBuffer>& gBuffers);
    protected:
        void createDescriptorManager(const PathTracerCreateInfo& info);
        void createPipelineLayout();
        void createShaders();
        void createComputePipelines();
        void createBuffers(const PathTracerCreateInfo& info);

        uint32_t mFramesInFlight = 2;
        uint32_t mInstanceCount = 1;

        Context* mContext = nullptr;
        DescriptorManager mDescriptorManager{};
        PipelineLayout mPipelineLayout{};
        ShaderModule mShader{};
        ComputePipelines mComputePipelines{};
        std::vector<Buffer> mCameraBuffers{};
        Buffer mTriangleBuffer{};
        Buffer mTriangleExtraBuffer{};
        Buffer mNodeBuffer{};
        Buffer mTLASNodeBuffer{};
        Buffer mInstanceBuffer{};
        std::vector<Buffer> mDirectLightBuffers{};
        std::vector<TexturesByType>* mTextures = nullptr;
        VkImageView mOutImageView = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_PATHTRACER_HPP