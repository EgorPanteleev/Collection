//
// Created by igor on 5/11/26.
//

#include "Outliner.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    Outliner::Outliner(const OutlinerCreateInfo& info): mContext(info.context), mFramesInFlight(info.framesInFlight) {
        createDescriptorManager(info);
        createPipelineLayout();
        createShaders();
        createGraphicsPipelines();
    }

    void Outliner::update(const OutlinerUpdateInfo& info) {
    }

    void Outliner::record(const OutlinerRecordInfo& info) {
        std::vector<VkClearValue> clearValues = {
        {.color = {{0.2f, 0.2f, 0.2f, 1.0f}},},
        {.depthStencil = {1.0f, 0}}
        };
        VkRenderingAttachmentInfo imageAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView =  info.outImageView,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = clearValues[0],
        };
        std::vector colorAttachments = {imageAttachment};

        const VkRenderingInfo renderingInfo = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {
                        .offset = {0, 0},
                        .extent = info.extent
                },
                .layerCount = 1,
                .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
                .pColorAttachments = colorAttachments.data(),
                .pDepthAttachment = nullptr,
        };

        vkCmdBeginRendering(info.commandBuffer, &renderingInfo);

        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipelines[0]);

        VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(info.extent.width),
                .height = static_cast<float>(info.extent.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f
        };
        vkCmdSetViewport(info.commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{
                .offset = {0, 0},
                .extent = info.extent,
        };
        vkCmdSetScissor(info.commandBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayout.get(),
                                0, 1, &mDescriptorManager.set(info.currentFrame), 0, nullptr);
        vkCmdDraw(info.commandBuffer, 3, 1, 0, 0);
        vkCmdEndRendering(info.commandBuffer);
    }

    void Outliner::createDescriptorManager(const OutlinerCreateInfo& info) {
        //layout
        constexpr BindingDescription samplerImageBinding {
            .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        mDescriptorManager.add(samplerImageBinding);
        mDescriptorManager.add(samplerImageBinding);

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
        };
        mDescriptorManager.build(buildInfo);

        //resources
        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager.add(i, ImageResource(info.tracerSampler, info.tracerImageView,
                VK_IMAGE_LAYOUT_GENERAL));
            mDescriptorManager.add(i, ImageResource(info.instanceIdSamplers[i], info.instanceIdImageViews[i],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        }
        mDescriptorManager.update();
    }

    void Outliner::createPipelineLayout() {
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void Outliner::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
            .fileName = COMPILED_SHADERS_DIR"/outliner.vert.spv"
        };
        mVertexShader = ShaderModule(createInfo);
        createInfo.fileName = COMPILED_SHADERS_DIR"/outliner.frag.spv";
        mFragmentShader = ShaderModule(createInfo);
    }

    void Outliner::createGraphicsPipelines() {
        std::vector<VkPipelineShaderStageCreateInfo> stages = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = mVertexShader.get(),
                .pName = "main",
                .pSpecializationInfo = nullptr
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = mFragmentShader.get(),
                .pName = "main",
                .pSpecializationInfo = nullptr
            }
        };

        const GraphicsPipelineCreateInfo createInfo {
            .device = mContext->device(),
            .layout = mPipelineLayout.get(),
            .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM},
            .bindingDescription = {},
            .attributeDescriptions = {},
            .stages = stages,
        };
        mGraphicsPipelines = GraphicsPipelines(createInfo);
    }
}