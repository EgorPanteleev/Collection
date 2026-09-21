//
// Created by igor on 6/9/26.
//

#include "View/PostprocessPass.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    static constexpr uint32_t HISTOGRAM_BINS     = 256;
    static constexpr uint32_t EXPOSURE_SLOT      = 256;
    static constexpr uint32_t EXPOSURE_WORDS     = 258;
    static constexpr uint32_t TONEMAP_PIPELINE   = 0;
    static constexpr uint32_t HISTOGRAM_PIPELINE = 1;
    static constexpr uint32_t EXPOSURE_PIPELINE  = 2;

    static void computeBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 srcStage,
                               VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                               VkAccessFlags2 dstAccess) {
        const VkMemoryBarrier2 barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = dstStage,
            .dstAccessMask = dstAccess
        };
        const VkDependencyInfo dependency {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier
        };
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    PostprocessPass::PostprocessPass(const PostprocessPassCreateInfo& info):
    mFramesInFlight(info.framesInFlight), mContext(info.context), mTracerView(info.tracerView),
    mInstanceView(info.instanceView), mOutputView(info.outputView) {
        createExposureBuffer();
        createDescriptorManager();
        createPipelineLayout();
        createShaders();
        createComputePipelines();
    }

    void PostprocessPass::update(const PostprocessPassUpdateInfo& info) {
    }

    ExposureReadback PostprocessPass::exposure() const {
        ExposureReadback result{};
        if (mExposureReadback.get() == VK_NULL_HANDLE) return result;
        float* data = nullptr;
        vmaMapMemory(mContext->allocator(), mExposureReadback.allocation(), reinterpret_cast<void**>(&data));
        result.exposure     = data[0];
        result.avgLuminance = data[1];
        vmaUnmapMemory(mContext->allocator(), mExposureReadback.allocation());
        return result;
    }

    void PostprocessPass::record(const PostprocessPassRecordInfo& info) {
        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineLayout.get(),
                        0, 1, &mDescriptorManager.set(info.currentFrame), 0, nullptr);
        vkCmdPushConstants(info.commandBuffer, mPipelineLayout.get(), VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(PostprocessPushConstants), &info.constants);

        if (info.constants.autoExposure != 0u) recordExposure(info);

        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelines[TONEMAP_PIPELINE]);
        auto [width, height]  = info.extent;
        vkCmdDispatch(info.commandBuffer, 1 + (width - 1) / 16, 1 + (height - 1) / 16, 1);
    }

    void PostprocessPass::recordExposure(const PostprocessPassRecordInfo& info) {
        vkCmdFillBuffer(info.commandBuffer, mExposureBuffer.get(), 0,
                        HISTOGRAM_BINS * sizeof(uint32_t), 0);
        computeBarrier(info.commandBuffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelines[HISTOGRAM_PIPELINE]);
        vkCmdDispatch(info.commandBuffer, 1 + (info.constants.renderWidth  - 1) / 16,
                                          1 + (info.constants.renderHeight - 1) / 16, 1);
        computeBarrier(info.commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelines[EXPOSURE_PIPELINE]);
        vkCmdDispatch(info.commandBuffer, 1, 1, 1);
        computeBarrier(info.commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);

        const VkBufferCopy copy {
            .srcOffset = EXPOSURE_SLOT * sizeof(uint32_t),
            .dstOffset = 0,
            .size = 2 * sizeof(uint32_t)
        };
        vkCmdCopyBuffer(info.commandBuffer, mExposureBuffer.get(), mExposureReadback.get(), 1, &copy);
    }

    void PostprocessPass::createExposureBuffer() {
        const BufferCreateInfo exposureInfo {
            .allocator = mContext->allocator(),
            .size = EXPOSURE_WORDS * sizeof(uint32_t),
            .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        mExposureBuffer = Buffer(exposureInfo);

        const BufferCreateInfo readbackInfo {
            .allocator = mContext->allocator(),
            .size = 2 * sizeof(uint32_t),
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        mExposureReadback = Buffer(readbackInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(),
                                            mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
        vkCmdFillBuffer(commandBuffer, mExposureBuffer.get(), 0, EXPOSURE_WORDS * sizeof(uint32_t), 0);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void PostprocessPass::createDescriptorManager() {
        mDescriptorManager
            .add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
            .add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
            .add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
            .add(BindingType::SSBO         , VK_SHADER_STAGE_COMPUTE_BIT)
            .build(mContext, mFramesInFlight);

        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager
                .bind(i, ImageResource(mTracerView, VK_IMAGE_LAYOUT_GENERAL))
                .bind(i, ImageResource(mInstanceView, VK_IMAGE_LAYOUT_GENERAL))
                .bind(i, ImageResource(mOutputView, VK_IMAGE_LAYOUT_GENERAL))
                .bind(i, BufferResource(mExposureBuffer));
        }
        mDescriptorManager.update();
    }

    void PostprocessPass::createPipelineLayout() {
        const VkPushConstantRange pushRange {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PostprocessPushConstants)
        };
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
            .ranges = {pushRange}
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void PostprocessPass::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
        };
        createInfo.fileName = COMPILED_SHADERS_DIR"/outline.slang.spv";
        mOutlineShader = ShaderModule(createInfo);
        createInfo.fileName = COMPILED_SHADERS_DIR"/exposure.slang.spv";
        mExposureShader = ShaderModule(createInfo);
    }

    void PostprocessPass::createComputePipelines() {
        const std::vector<VkPipelineShaderStageCreateInfo> stages {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = mOutlineShader.get(),
                    .pName = "main",
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = mExposureShader.get(),
                    .pName = "histogramMain",
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = mExposureShader.get(),
                    .pName = "exposureMain",
                },
        };
        const ComputePipelinesCreateInfo createInfo {
            .device = mContext->device(),
            .stages = stages,
            .layouts = {mPipelineLayout.get(), mPipelineLayout.get(), mPipelineLayout.get()},
        };
        mPipelines = ComputePipelines(createInfo);
    }

}
