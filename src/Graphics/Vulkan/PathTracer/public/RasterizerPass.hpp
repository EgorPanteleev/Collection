//
// Created by igor on 6/9/26.
//

#ifndef COLLECTION_RasterizerPassPASS_HPP
#define COLLECTION_RasterizerPassPASS_HPP

#include "Context.hpp"
#include "DescriptorManager.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "GraphicsPipelines.hpp"
#include "Types.hpp"
#include "Camera.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    struct RasterizerPassCreateInfo {
        Context*                   context        = nullptr;
        ImageView*                 outView        = nullptr;
        VkFormat                   outFormat      = VK_FORMAT_UNDEFINED;
        VkExtent2D                 extent{};
        uint32_t                   framesInFlight = 2;
    };

    struct RasterizerPassUpdateInfo {
        cs::AbsCamera* camera       = nullptr;
        InstanceData*  instance     = nullptr;
        uint32_t       currentFrame = 0;
    };

    struct RasterizerPassRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        Buffer*         vertexBuffer  = nullptr;
        Buffer*         indexBuffer   = nullptr;
        uint32_t        indexCount    = 0;
        VkExtent2D      extent{};
        uint32_t        currentFrame  = 0;
    };

    class RasterizerPass {
    public:
        RasterizerPass() = default;
        explicit RasterizerPass(const RasterizerPassCreateInfo& info);
        void update(const RasterizerPassUpdateInfo& info);
        void record(const RasterizerPassRecordInfo& info);
    protected:
        void createImages(VkExtent2D extent);
        void createDescriptorManager();
        void createPipelineLayout();
        void createShaders();
        void createGraphicsPipelines();
        void createBuffers();

        uint32_t                   mFramesInFlight    = 1;

        Context*                   mContext           = nullptr;
        DescriptorManager          mDescriptorManager{};
        PipelineLayout             mPipelineLayout    = CRV_NULL_HANDLE;
        GraphicsPipelines          mGraphicsPipelines = CRV_NULL_HANDLE;

        ShaderModule               mVertexShader      = CRV_NULL_HANDLE;
        ShaderModule               mFragmentShader    = CRV_NULL_HANDLE;

        ImageView*                 mOutputView         = nullptr;
        Image                      mDepthImage         = CRV_NULL_HANDLE;
        ImageView                  mDepthView          = CRV_NULL_HANDLE;

        VkFormat                   mOutputFormat       = VK_FORMAT_UNDEFINED;

        std::vector<Buffer>        mMVPBuffers{};
        std::vector<Buffer>        mInstanceBuffers{};
    };
}

#endif //COLLECTION_RasterizerPassPASS_HPP