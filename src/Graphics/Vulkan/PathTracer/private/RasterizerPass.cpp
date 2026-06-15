//
// Created by igor on 6/9/26.
//

#include "RasterizerPass.hpp"

namespace crv::graphics::vulkan {
    RasterizerPass::RasterizerPass(const RasterizerPassCreateInfo& info):
    mFramesInFlight(info.framesInFlight), mContext(info.context),
    mOutputView(info.outView), mOutputFormat(info.outFormat) {
        createImages(info.extent);
        createBuffers();
        createDescriptorManager();
        createPipelineLayout();
        createShaders();
        createGraphicsPipelines();
    }

    void RasterizerPass::update(const RasterizerPassUpdateInfo& info) {
        const MVPGPU MVP {
            .model = glm::mat4(1.0f),
            .view = info.camera->viewMatrix(),
            .proj = info.camera->projectionMatrix(),
        };
        copyDataToBuffer(mContext, QueueFamilyType::GRAPHICS, &MVP, sizeof(MVPGPU), mMVPBuffers[info.currentFrame]);

        const glm::mat4 instanceModel = info.instance->transform.matrix();
        copyDataToBuffer(mContext, QueueFamilyType::GRAPHICS, &instanceModel, sizeof(glm::mat4),
            mInstanceBuffers[info.currentFrame]);
    }

    void RasterizerPass::record(const RasterizerPassRecordInfo& info) {
        const VkRenderingAttachmentInfo instanceAttachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = mOutputView->get(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.uint32 = {0, 0, 0, 0}}},
        };
        std::vector colorAttachments = {instanceAttachment};
        const VkRenderingAttachmentInfo depthAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = mDepthView.get(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .clearValue = {.depthStencil = {1.0f, 0}},
        };
        const VkRenderingInfo renderingInfo = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {
                        .offset = {0, 0},
                        .extent = info.extent
                },
                .layerCount = 1,
                .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
                .pColorAttachments = colorAttachments.data(),
                .pDepthAttachment = &depthAttachment,
        };

        vkCmdBeginRendering(info.commandBuffer, &renderingInfo);
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipelines[0]);

        VkViewport viewport = getDefaultViewport(info.extent);
        vkCmdSetViewport(info.commandBuffer, 0, 1, &viewport);
        VkRect2D scissor = getDefaultScissor(info.extent);
        vkCmdSetScissor(info.commandBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayout.get(),
                                0, 1, &mDescriptorManager.set(info.currentFrame), 0, nullptr);
        if (info.vertexBuffer and info.indexBuffer) {
            VkBuffer vertexBuffers[] = { info.vertexBuffer->get() };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(info.commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(info.commandBuffer, info.indexBuffer->get(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(info.commandBuffer, info.indexCount, 1,
      0, 0, 0);
        }
        vkCmdEndRendering(info.commandBuffer);
    }

    void RasterizerPass::createImages(VkExtent2D extent) {
        const ImageCreateInfo depthImageCreateInfo {
            .device = mContext->device(),
            .allocator = mContext->allocator(),
            .format = VK_FORMAT_D32_SFLOAT,
            .extent = {extent.width, extent.height, 1},
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        mDepthImage = Image(depthImageCreateInfo);
        const ImageViewCreateInfo depthViewCreateInfo {
            .device = mContext->device(),
            .image = mDepthImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        };
        mDepthView = ImageView(depthViewCreateInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(), mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
        const ImageTransitInfo2 transitInfo {
            .commandBuffer = commandBuffer,
            .image = mDepthImage.get(),
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        };
        Image::transit(transitInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void RasterizerPass::createDescriptorManager() {
        mDescriptorManager.add(BindingType::UBO    , VK_SHADER_STAGE_VERTEX_BIT  );
        mDescriptorManager.add(BindingType::UBO    , VK_SHADER_STAGE_VERTEX_BIT  );

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
            .variableCount = 0
        };
        mDescriptorManager.build(buildInfo);

        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager.add(i, BufferResource(mMVPBuffers[i]  ));
            mDescriptorManager.add(i, BufferResource(mInstanceBuffers[i] ));
        }
        mDescriptorManager.update();
    }

    void RasterizerPass::createPipelineLayout() {
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void RasterizerPass::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
        };
        createInfo.fileName = COMPILED_SHADERS_DIR"/rasterizer.slang.spv";
        mVertexShader = ShaderModule(createInfo);
        mFragmentShader = ShaderModule(createInfo);
    }

    void RasterizerPass::createGraphicsPipelines() {
        constexpr VkVertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };
        constexpr VkVertexInputAttributeDescription desc1 {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, pos)
        };
        constexpr VkVertexInputAttributeDescription desc2 {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, texCoord)
        };
        constexpr VkVertexInputAttributeDescription desc3 {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal)
        };
        constexpr VkVertexInputAttributeDescription desc4 {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(Vertex, tangent)
        };
        const std::vector attributeDescriptions{desc1, desc2, desc3, desc4};

        const std::vector<VkPipelineShaderStageCreateInfo> stages = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = mVertexShader.get(),
                .pName = "vertMain",
                .pSpecializationInfo = nullptr
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = mFragmentShader.get(),
                .pName = "fragMain",
                .pSpecializationInfo = nullptr
            }
        };

        const GraphicsPipelineCreateInfo passInfo {
            .device = mContext->device(),
            .layout = mPipelineLayout.get(),
            .colorFormats = {mOutputFormat},
            .bindingDescription = bindingDescription,
            .attributeDescriptions = attributeDescriptions,
            .stages = stages,
        };
        mGraphicsPipelines = GraphicsPipelines(passInfo);
    }

    void RasterizerPass::createBuffers() {
        mMVPBuffers.resize(mFramesInFlight);
        mInstanceBuffers.resize(mFramesInFlight);
        UBOData uboData{};
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            uboData.add<MVPGPU>(mMVPBuffers[i]);
            uboData.add<glm::mat4>(mInstanceBuffers[i]);
        }
        uboData.createAll(mContext);
    }
}
