//
// Created by igor on 6/9/26.
//

#ifndef COLLECTION_POSTPROCESSPASS_HPP
#define COLLECTION_POSTPROCESSPASS_HPP

#include "Context.hpp"
#include "DescriptorManager.hpp"
#include "PipelineLayout.hpp"
#include "ComputePipelines.hpp"
#include "ShaderModule.hpp"
#include "Buffer.hpp"
#include "View/Types.hpp"

namespace crv::graphics::vulkan {
    struct PostprocessPassCreateInfo {
        Context*    context         = nullptr;
        ImageView*  tracerView      = nullptr;
        ImageView*  instanceView    = nullptr;
        ImageView*  outputView      = nullptr;
        uint32_t    framesInFlight  = 0;
    };

    struct PostprocessPassUpdateInfo {
    };

    struct PostprocessPassRecordInfo {
        VkCommandBuffer          commandBuffer = VK_NULL_HANDLE;
        VkExtent2D               extent{};
        uint32_t                 currentFrame  = 0;
        PostprocessPushConstants constants{};
    };

    struct ExposureReadback {
        float exposure     = 0.0f;
        float avgLuminance = 0.0f;
    };

    class PostprocessPass {
    public:
        PostprocessPass() = default;
        explicit PostprocessPass(const PostprocessPassCreateInfo& info);
        void update(const PostprocessPassUpdateInfo& info);
        void record(const PostprocessPassRecordInfo& info);
        [[nodiscard]] ExposureReadback exposure() const;
    protected:
        void recordExposure(const PostprocessPassRecordInfo& info);
        void createExposureBuffer();
        void createDescriptorManager();
        void createPipelineLayout();
        void createShaders();
        void createComputePipelines();

        uint32_t          mFramesInFlight    = 1;

        Context*          mContext           = nullptr;
        DescriptorManager mDescriptorManager{};
        PipelineLayout    mPipelineLayout    = CRV_NULL_HANDLE;
        ShaderModule      mOutlineShader     = CRV_NULL_HANDLE;
        ShaderModule      mExposureShader    = CRV_NULL_HANDLE;
        ComputePipelines  mPipelines         = CRV_NULL_HANDLE;
        Buffer            mExposureBuffer    = CRV_NULL_HANDLE;
        Buffer            mExposureReadback  = CRV_NULL_HANDLE;

        ImageView*        mTracerView         = nullptr;
        ImageView*        mInstanceView       = nullptr;
        ImageView*        mOutputView         = nullptr;
    };
}

#endif //COLLECTION_POSTPROCESSPASS_HPP