//
// Created by igor on 4/26/26.
//

#include "Rasterizer.hpp"
#include "CoreUtils.hpp"
#include "Types.hpp"

namespace crv::graphics::vulkan {
    Rasterizer::Rasterizer(const RasterizerCreateInfo& info): mFramesInFlight(info.framesInFlight),
    mColorFormat(info.colorFormat), mNormalFormat(info.normalFormat), mInstanceIdFormat(info.instanceIdFormat),
    mContext(info.context), mTextures(info.textures), mMeshesData(info.meshesData) {
        createImages(info.extent);
        createBuffers(info);
        createDescriptorManager();
        createPipelineLayout();
        createShaders();
        createGraphicsPipelines();
    }

    void Rasterizer::update(const RasterizerUpdateInfo& info) {
        glm::mat4 model = glm::mat4(1.0f);
        AlignedMVP MVP {
            .model = model,
            .view = info.camera->viewMatrix(),
            .proj = info.camera->projectionMatrix(),
            .trInvModel = glm::transpose(glm::inverse(model))
        };
        copyDataToBuffer(mContext, QueueFamilyType::GRAPHICS, &MVP, sizeof(AlignedMVP), mMVPBuffers[info.currentFrame]);
    }

    void Rasterizer::record(const RasterizerRecordInfo& info) {
        recordMainPass(info);
        recordSelectedInstancePass(info);
        recordPixelRead(info);
    }

    void Rasterizer::updateSelectedInstance() {
        uint32_t* data = nullptr;
        vmaMapMemory(mContext->allocator(), mReadbackBuffer.allocation(), (void**)&data);
        mSelectedInstanceId = *data;
        vmaUnmapMemory(mContext->allocator(), mReadbackBuffer.allocation());
    }

    void Rasterizer::updateInstanceBuffer(const std::vector<MeshInstance>& instances) {
        std::vector<RasterInstance> rasterInstances;
        for (const auto& instance: instances) rasterInstances.emplace_back(instance);
        copyDataToBuffer(mContext, QueueFamilyType::GRAPHICS, rasterInstances.data(),
            rasterInstances.size() * sizeof(RasterInstance), mInstanceBuffer);
    }

    void Rasterizer::recordMainPass(const RasterizerRecordInfo& info) {
        std::vector<VkClearValue> clearValues = {
        {.color = {{0.2f, 0.2f, 0.2f, 1.0f}},},
        {.depthStencil = {1.0f, 0}}
        };
        VkRenderingAttachmentInfo albedoAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView =  info.gBuffer->colorView.get(),
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = clearValues[0],
        };
        VkRenderingAttachmentInfo normalAttachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView =  info.gBuffer->normalView.get(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clearValues[0],
        };
        VkRenderingAttachmentInfo instanceIdAttachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = mInstanceIdView.get(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.uint32 = {0, 0, 0, 0}}},
        };
        std::vector colorAttachments = {albedoAttachment, normalAttachment, instanceIdAttachment};
        VkRenderingAttachmentInfo depthAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = info.gBuffer->depthView.get(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = clearValues[1],
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
        VkBuffer vertexBuffers[] = { mVertexBuffer.get() };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(info.commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(info.commandBuffer, mIndexBuffer.get(),
                     0, VK_INDEX_TYPE_UINT32);

        for (const auto& meshData : mMeshesData) {
            vkCmdDrawIndexed(info.commandBuffer, meshData.indexCount, meshData.instanceCount,
                meshData.baseIndex, static_cast<int32_t>(meshData.baseVertex), meshData.baseInstance);
        }
        vkCmdEndRendering(info.commandBuffer);
    }

    void Rasterizer::recordSelectedInstancePass(const RasterizerRecordInfo& info) {
        VkRenderingAttachmentInfo selectedInstanceAttachment = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView =  info.gBuffer->selectedInstanceView.get(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {.uint32 = {0, 0, 0, 0}}},
        };
        const VkRenderingInfo renderingInfo = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {
                .offset = {0, 0},
                .extent = info.extent
            },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &selectedInstanceAttachment,
            .pDepthAttachment = nullptr,
        };

        vkCmdBeginRendering(info.commandBuffer, &renderingInfo);
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipelines[1]);

        VkViewport viewport = getDefaultViewport(info.extent);
        vkCmdSetViewport(info.commandBuffer, 0, 1, &viewport);
        VkRect2D scissor = getDefaultScissor(info.extent);
        vkCmdSetScissor(info.commandBuffer, 0, 1, &scissor);

        uint32_t selectedInstanceId = mSelectedInstanceId - 1;
        uint32_t instanceCount = mSelectedInstanceId ? 1 : 0;
        for (const auto& meshData: mMeshesData) {
            if (selectedInstanceId < meshData.baseInstance or
                selectedInstanceId >= meshData.baseInstance + meshData.instanceCount) continue;
            vkCmdDrawIndexed(info.commandBuffer, meshData.indexCount, instanceCount,
                meshData.baseIndex, static_cast<int32_t>(meshData.baseVertex), selectedInstanceId);
        }
        vkCmdEndRendering(info.commandBuffer);
    }

    void Rasterizer::recordPixelRead(const RasterizerRecordInfo& info) {
        if (info.clickPos.x == UINT32_MAX) return;
        const ImageBarrierInfo readbackBarrierInfo {
            .image = mInstanceIdImage.get(),
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        VkImageMemoryBarrier readbackBarrier = Image::barrier(readbackBarrierInfo);

        ImagePipelineBarrierInfo readbackPipelineBarrierInfo {
            .commandBuffer = info.commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT,
            .barriers = {readbackBarrier}
        };
        Image::pipelineBarrier(readbackPipelineBarrierInfo);

        VkBufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {static_cast<int32_t>(info.clickPos.x), static_cast<int32_t>(info.clickPos.y), 0},
            .imageExtent = {1, 1, 1}
        };
        vkCmdCopyImageToBuffer(info.commandBuffer,
            mInstanceIdImage.get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mReadbackBuffer.get(),
            1, &region);

        VkImageMemoryBarrier invReadbackBarrier = Image::inverseBarrier(readbackBarrierInfo);
        readbackPipelineBarrierInfo.barriers = {invReadbackBarrier};
        Image::inversePipelineBarrier(readbackPipelineBarrierInfo);
    }

    void Rasterizer::createImages(VkExtent3D extent) {
        const ImageCreateInfo imageCreateInfo {
            .device = mContext->device(),
            .allocator = mContext->allocator(),
            .flags = 0,
            .format = VK_FORMAT_R32_UINT,
            .extent = extent,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        mInstanceIdImage = Image(imageCreateInfo);

        const ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext->device(),
            .image = mInstanceIdImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R32_UINT,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        mInstanceIdView = ImageView(imageViewCreateInfo);

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext->device(), mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const ImageTransitInfo instanceIdTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mInstanceIdImage.get(),
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        Image::transit(instanceIdTransitInfo);
        endCommandBuffer(commandPool, commandBuffers, mContext->queue(QueueFamilyType::GRAPHICS));

        #ifndef NDEBUG
            DEBUG << "Instance id image: " << mInstanceIdImage.get();
        #endif
    }

    void Rasterizer::createDescriptorManager() {
        //layout
        constexpr BindingDescription UBOBinding {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages = VK_SHADER_STAGE_VERTEX_BIT,
        };
        constexpr BindingDescription SSBOBinding {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_VERTEX_BIT,
        };
        const BindingDescription texturesBinding {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
            .flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
            .count = static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)
        };

        mDescriptorManager.add(UBOBinding );
        mDescriptorManager.add(SSBOBinding);
        mDescriptorManager.add(texturesBinding );

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
            .variableCount = static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)
        };
        mDescriptorManager.build(buildInfo);

        //resources
        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager.add(i, BufferResource(mMVPBuffers[i]  ));
            mDescriptorManager.add(i, BufferResource(mInstanceBuffer ));
            mDescriptorManager.add(i, ImageResource(*mTextures));
        }
        mDescriptorManager.update();
    }

    void Rasterizer::createPipelineLayout() {
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void Rasterizer::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
            .fileName = COMPILED_SHADERS_DIR"/rasterizer.vert.spv"
        };
        mVertexShader = ShaderModule(createInfo);
        createInfo.fileName = COMPILED_SHADERS_DIR"/rasterizer.frag.spv";
        mFragmentShader = ShaderModule(createInfo);
        createInfo.fileName = COMPILED_SHADERS_DIR"/rasterizerSelected.frag.spv";
        mSelectedFragmentShader = ShaderModule(createInfo);
    }

    void Rasterizer::createGraphicsPipelines() {
        const VkVertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };
        const VkVertexInputAttributeDescription desc1 {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, pos)
        };
        const VkVertexInputAttributeDescription desc2 {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, texCoord)
        };
        const VkVertexInputAttributeDescription desc3 {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal)
        };
        const VkVertexInputAttributeDescription desc4 {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(Vertex, tangent)
        };
        std::vector attributeDescriptions{desc1, desc2, desc3, desc4};

        std::vector<VkPipelineShaderStageCreateInfo> pass1Stages = {
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

        std::vector<VkPipelineShaderStageCreateInfo> pass2Stages = {
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
                .module = mSelectedFragmentShader.get(),
                .pName = "main",
                .pSpecializationInfo = nullptr
            }
        };

        const GraphicsPipelineCreateInfo pass1Info {
            .device = mContext->device(),
            .layout = mPipelineLayout.get(),
            .colorFormats = {mColorFormat, mNormalFormat, mInstanceIdFormat},
            .bindingDescription = bindingDescription,
            .attributeDescriptions = attributeDescriptions,
            .stages = pass1Stages,
        };

        const GraphicsPipelineCreateInfo pass2Info {
            .device = mContext->device(),
            .layout = mPipelineLayout.get(),
            .colorFormats = {mInstanceIdFormat},
            .bindingDescription = bindingDescription,
            .attributeDescriptions = attributeDescriptions,
            .stages = pass2Stages,
        };
        mGraphicsPipelines = GraphicsPipelines({pass1Info, pass2Info});
    }

    void Rasterizer::createBuffers(const RasterizerCreateInfo &info) {
        {
            const size_t verticesSize = sizeof(Vertex) * info.vertices.size();
            const BufferCreateInfo vertexBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = verticesSize,
                .bufferUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mVertexBuffer = Buffer(vertexBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<Vertex*>(info.vertices.data()),
                .size = verticesSize,
                .allocator = mContext->allocator(),
                .buffer = mVertexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }

        {
            const size_t indicesSize = sizeof(uint32_t) * info.indices.size();
            const BufferCreateInfo indexBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = indicesSize,
                .bufferUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mIndexBuffer = Buffer(indexBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<uint32_t*>(info.indices.data()),
                .size = indicesSize,
                .allocator = mContext->allocator(),
                .buffer = mIndexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }

        std::vector<RasterInstance> instances;
        for (auto instance: info.instances) instances.emplace_back(instance);
        SSBOData ssboData{};
        ssboData.add(instances     , mInstanceBuffer);
        ssboData.createAll(mContext, QueueFamilyType::GRAPHICS);

        mMVPBuffers.resize(mFramesInFlight);
        UBOData uboData{};
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            uboData.add<AlignedMVP     >(mMVPBuffers[i]     );
        }
        uboData.createAll(mContext, QueueFamilyType::GRAPHICS);

        const BufferCreateInfo readbackInfo {
            .allocator = mContext->allocator(),
            .size = sizeof(uint32_t),
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        mReadbackBuffer = Buffer(readbackInfo);

        uint32_t data = 0;
        const CopyDataToCPUBufferInfo copyInfo {
            .data = &data,
            .size = sizeof(uint32_t),
            .allocator = mContext->allocator(),
            .allocation = mReadbackBuffer.allocation()
        };
        Buffer::copy(copyInfo);
    }
}
