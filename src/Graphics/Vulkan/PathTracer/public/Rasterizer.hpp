//
// Created by igor on 4/26/26.
//

#ifndef COLLECTION_RASTERPASS_HPP
#define COLLECTION_RASTERPASS_HPP

#include "Context.hpp"
#include "Types.hpp"
#include "Image.hpp"
#include "DescriptorSetLayout.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSets.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "GraphicsPipelines.hpp"
#include "Buffer.hpp"
#include "AbsCamera.hpp"

namespace crv::graphics::vulkan {
    struct RasterizerCreateInfo {
        Context*                     context        = nullptr;
        VkFormat                     colorFormat    = VK_FORMAT_UNDEFINED;
        VkFormat                     normalFormat   = VK_FORMAT_UNDEFINED;
        VkExtent3D                   extent{};
        uint32_t                     framesInFlight = 2;
        std::vector<MeshData>        meshesData{};
        std::vector<MeshInstance>    instances{};
        std::vector<TexturesByType>* textures       = nullptr;
    };

    struct RasterizerUpdateInfo {
        scene::AbsCamera* camera       = nullptr;
        uint32_t          currentFrame = 0;
    };

    struct RasterizerRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        GBuffer*        gBuffer       = nullptr;
        VkExtent2D      extent{};
        uint32_t        currentFrame  = 0;
    };

    struct MeshBuffer {
        Buffer vertexBuffer{};
        Buffer indexBuffer{};
        uint32_t indexCount = 0;
        uint32_t firstInstance = 0;
        uint32_t instanceCount = 0;
    };

    class Rasterizer {
    public:
        Rasterizer() = default;
        explicit Rasterizer(const RasterizerCreateInfo& info);
        void update(const RasterizerUpdateInfo& info);
        void record(const RasterizerRecordInfo& info);
    protected:
        void createDescriptorSetLayout();
        void createDescriptorPool();
        std::vector<VkDescriptorSetLayout>  getDescriptorLayouts();
        void createDescriptorSets();
        void createPipelineLayout();
        void createShaders();
        void createGraphicsPipelines();
        void createBuffers(const RasterizerCreateInfo& info);

        uint32_t mFramesInFlight = 1;
        VkFormat mColorFormat  = VK_FORMAT_UNDEFINED;
        VkFormat mNormalFormat = VK_FORMAT_UNDEFINED;

        Context* mContext = nullptr;
        DescriptorSetLayout mDescriptorSetLayout{};
        DescriptorPool mDescriptorPool{};
        DescriptorSets mDescriptorSets{};
        std::vector<TexturesByType>* mTextures = nullptr;
        PipelineLayout mPipelineLayout{};
        ShaderModule mVertexShader{};
        ShaderModule mFragmentShader{};
        GraphicsPipelines mGraphicsPipelines{};
        std::vector<Buffer> mMVPBuffers{};
        std::vector<MeshBuffer> mMeshBuffers{};
        Buffer mInstanceBuffer{};
    };
}

#endif //COLLECTION_RASTERPASS_HPP