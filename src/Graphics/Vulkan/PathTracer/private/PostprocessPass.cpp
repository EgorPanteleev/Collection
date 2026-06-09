//
// Created by igor on 6/9/26.
//

#include "PostprocessPass.hpp"

namespace crv::graphics::vulkan {
    PostprocessPass::PostprocessPass(const PostprocessPassCreateInfo& info):
    mFramesInFlight(info.framesInFlight), mContext(info.context), mTracerView(info.tracerView),
    mInstanceView(info.instanceView), mOutputView(info.outputView) {
        createDescriptorManager();
        createPipelineLayout();
        createShaders();
        createComputePipelines();
    }

    void PostprocessPass::update(const PostprocessPassUpdateInfo& info) {
    }

    void PostprocessPass::record(const PostprocessPassRecordInfo& info) {
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelines[0]);
        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineLayout.get(),
                        0, 1, &mDescriptorManager.set(info.currentFrame), 0, nullptr);

        auto [width, height]  = info.extent;
        const uint32_t groupX = 1 + (width  - 1) / 16;
        const uint32_t groupY = 1 + (height - 1) / 16;
        vkCmdDispatch(info.commandBuffer, groupX, groupY, 1);
    }

    void PostprocessPass::createDescriptorManager() {
        mDescriptorManager.add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);
        mDescriptorManager.add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);
        mDescriptorManager.add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
        };
        mDescriptorManager.build(buildInfo);

        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager.add(i, ImageResource(mTracerView, VK_IMAGE_LAYOUT_GENERAL));
            mDescriptorManager.add(i, ImageResource(mInstanceView, VK_IMAGE_LAYOUT_GENERAL));
            mDescriptorManager.add(i, ImageResource(mOutputView, VK_IMAGE_LAYOUT_GENERAL));
        }
        mDescriptorManager.update();
    }

    void PostprocessPass::createPipelineLayout() {
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
            .ranges = {}
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void PostprocessPass::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
        };
        createInfo.fileName = COMPILED_SHADERS_DIR"/outline.comp.spv";
        mOutlineShader = ShaderModule(createInfo);
    }

    void PostprocessPass::createComputePipelines() {
        const std::vector<VkPipelineShaderStageCreateInfo> stages {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = mOutlineShader.get(),
                    .pName = "main",
                    .pSpecializationInfo = nullptr
                },
        };
        const ComputePipelinesCreateInfo createInfo {
            .device = mContext->device(),
            .stages = stages,
            .layouts = {mPipelineLayout.get()},
        };
        mPipelines = ComputePipelines(createInfo);
    }

}