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
#include "Buffer.hpp"
#include "GPUTypes.hpp"

namespace crv::scene {
    class AbsCamera;
}

namespace crv::graphics::vulkan {
    struct PathTracerCreateInfo {
        Context* context;
        std::vector<AlignedTriangle> triangles{};
        std::vector<AlignedTriangleExtra> triangleExtras{};
        std::vector<AlignedNode> nodes{};
        std::vector<TexturesByType>* textures = nullptr;
        std::vector<uint32_t> materialIndices{};
        uint32_t framesInFlight = 2;
    };

    struct PathTracerUpdateInfo {
        scene::AbsCamera*  camera           = nullptr;
        AlignedDirectLight directLight{};
        VkImage            presentImage     = VK_NULL_HANDLE;
        VkImageView        presentImageView = VK_NULL_HANDLE;
        uint32_t           currentFrame     = 0;
    };

    struct PathTracerRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkExtent2D      extent{};
        uint32_t        currentFrame     = 0;
        uint32_t        frameCount       = 0;
        uint32_t        maxDepth         = 1;
    };

    class PathTracer {
    public:
        PathTracer() = default;
        explicit PathTracer(const PathTracerCreateInfo& info);
        void update(const PathTracerUpdateInfo& info);
        void record(const PathTracerRecordInfo& info);
    protected:
        void createDescriptorSetLayout();
        void createDescriptorPool();
        void createDescriptorSets();
        void createPipelineLayout();
        void createShaders();
        void createComputePipelines();
        void createBuffers(const PathTracerCreateInfo& info);

        [[nodiscard]] std::vector<VkDescriptorSetLayout> getDescriptorLayouts() const;
        uint32_t mFramesInFlight = 2;

        Context* mContext = nullptr;
        DescriptorSetLayout mDescriptorSetLayout{};
        DescriptorPool mDescriptorPool{};
        DescriptorSets mDescriptorSets{};
        PipelineLayout mPipelineLayout{};
        ShaderModule mShader{};
        ComputePipelines mComputePipelines{};
        std::vector<Buffer> mCameraBuffers{};
        Buffer mTriangleBuffer{};
        Buffer mTriangleExtraBuffer{};
        Buffer mNodeBuffer{};
        Buffer mMaterialIndexBuffer{};
        std::vector<Buffer> mDirectLightBuffers{};
        std::vector<TexturesByType>* mTextures = nullptr;
    };
}

#endif //COLLECTION_PATHTRACER_HPP