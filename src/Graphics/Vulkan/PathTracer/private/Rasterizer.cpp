//
// Created by igor on 4/26/26.
//

#include "Rasterizer.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    Rasterizer::Rasterizer(const RasterizerCreateInfo& info): mFramesInFlight(info.framesInFlight), mColorFormat(info.colorFormat),
    mNormalFormat(info.normalFormat), mContext(info.context), mTextures(info.textures) {
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets();
        createPipelineLayout();
        createShaders();
        createGraphicsPipelines();
        createBuffers(info);
    }

    void Rasterizer::update(const RasterizerUpdateInfo& info) {
        Buffer& MVPBuffer = mMVPBuffers[info.currentFrame];
        {
            glm::mat4 model = glm::mat4(1.0f);
            AlignedMVP MVP {
                .model = model,
                .view = info.camera->viewMatrix(),
                .proj = info.camera->projectionMatrix(),
                .trInvModel = glm::transpose(glm::inverse(model))
            };
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = &MVP,
                .size = sizeof(AlignedMVP),
                .allocator = mContext->allocator(),
                .buffer = MVPBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }

        VkDescriptorBufferInfo MVPBufferInfo {
            .buffer = MVPBuffer.get(),
            .offset = 0,
            .range = MVPBuffer.size()
        };

        VkDescriptorBufferInfo instanceBufferInfo {
            .buffer = mInstanceBuffer.get(),
            .offset = 0,
            .range = mInstanceBuffer.size()
        };

        std::vector<VkDescriptorImageInfo> textureInfos;
        for (size_t i = 0; i < mTextures->size(); i++) {
            auto& texturesByType = (*mTextures)[i];
            for (int j = 0; j < cm::Texture::Type::UNKNOWN; ++j) {
                auto& texture = texturesByType[j];
                VkDescriptorImageInfo textureInfo {
                    .sampler = texture.sampler(),
                    .imageView = texture.view(),
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                textureInfos.push_back(textureInfo);
            }
        }

        const VkWriteDescriptorSet writeDescriptorSet0 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &MVPBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet1 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &instanceBufferInfo
        };

        VkWriteDescriptorSet writeDescriptorSet2 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = textureInfos.data()
        };

        std::vector descriptorWrites{
            writeDescriptorSet0, writeDescriptorSet1,
            writeDescriptorSet2
        };

        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            descriptorsWrites.push_back(descriptorWrites);
        }

        const DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
    }

    void Rasterizer::record(const RasterizerRecordInfo& info) {
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

        std::vector colorAttachments = {albedoAttachment, normalAttachment};

        VkRenderingAttachmentInfo depthAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = info.gBuffer->depthView.get(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
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
                                0, 1, &mDescriptorSets[info.currentFrame], 0, nullptr);

        for (const auto& meshBuffer : mMeshBuffers) {
            VkBuffer vertexBuffers[] = { meshBuffer.vertexBuffer.get() };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(info.commandBuffer, 0, 1, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(info.commandBuffer, meshBuffer.indexBuffer.get(),
                                 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(info.commandBuffer,
                             meshBuffer.indexCount,
                             meshBuffer.instanceCount,
                             0, 0, meshBuffer.firstInstance);
        }

        vkCmdEndRendering(info.commandBuffer);
    }

    void Rasterizer::createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding binding0{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = nullptr
        };

        VkDescriptorSetLayoutBinding binding1{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = nullptr
        };

        VkDescriptorSetLayoutBinding binding2 {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN),
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
        };

        std::vector bindings = {binding0, binding1, binding2};

        std::vector<VkDescriptorBindingFlags> bindingFlags = {
            0, 0,  VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
        };

        const DescriptorSetLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .bindings = bindings,
            .bindingFlags = bindingFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }

    void Rasterizer::createDescriptorPool() {
        std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER , mFramesInFlight *
                static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)}
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext->device(),
            .flags = 0,
            .poolSizes = poolSizes,
            .maxSets = mFramesInFlight
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    std::vector<VkDescriptorSetLayout>  Rasterizer::getDescriptorLayouts() {
        std::vector<VkDescriptorSetLayout> layouts;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            layouts.push_back(mDescriptorSetLayout.get());
        }
        return layouts;
    }

    void Rasterizer::createDescriptorSets() {
        const std::vector variableCounts(mFramesInFlight, static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN));
        const DescriptorSetsCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = getDescriptorLayouts(),
            .pool = mDescriptorPool.get(),
            .variableCounts = variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);
    }

    void Rasterizer::createPipelineLayout() {
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = getDescriptorLayouts(),
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
        const VkVertexInputAttributeDescription desc5 {
            .location = 4,
            .binding = 0,
            .format = VK_FORMAT_R32_UINT,
            .offset = offsetof(Vertex, texIndex)
        };
        std::vector attributeDescriptions{desc1, desc2, desc3, desc4, desc5};

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

        const GraphicsPipelinesCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = {mPipelineLayout.get()},
            .colorFormats = {mColorFormat, mNormalFormat},
            .bindingDescription = bindingDescription,
            .attributeDescriptions = attributeDescriptions,
            .stages = stages,
        };
        mGraphicsPipelines = GraphicsPipelines(createInfo);
    }

    void Rasterizer::createBuffers(const RasterizerCreateInfo &info) {
        mMVPBuffers.resize(mFramesInFlight);
        const BufferCreateInfo MVPBufferCreateInfo {
            .allocator = mContext->allocator(),
            .size = sizeof(AlignedMVP),
            .bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mMVPBuffers[i] = Buffer(MVPBufferCreateInfo);
        }
        mMeshBuffers.resize(info.meshesData.size());
        std::vector<glm::mat4> instanceModels;
        for (size_t i = 0; i < info.meshesData.size(); ++i) {
            auto& meshBuffer = mMeshBuffers[i];
            auto& meshData = info.meshesData[i];
            {
                const size_t verticesSize = sizeof(Vertex) * meshData.vertices.size();
                const BufferCreateInfo vertexBufferCreateInfo {
                    .allocator = mContext->allocator(),
                    .size = verticesSize,
                    .bufferUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
                };
                meshBuffer.vertexBuffer = Buffer(vertexBufferCreateInfo);
                const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                    .data = const_cast<Vertex*>(meshData.vertices.data()),
                    .size = verticesSize,
                    .allocator = mContext->allocator(),
                    .buffer = meshBuffer.vertexBuffer.get(),
                    .device = mContext->device(),
                    .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                    .queue = mContext->queue(QueueFamilyType::GRAPHICS)
                };
                Buffer::copy(copyDataToGPUBufferInfo);
            }
            {
                meshBuffer.indexCount = meshData.indices.size();
                meshBuffer.instanceCount = meshData.instances.size();
                const size_t indicesSize = sizeof(uint32_t) * meshData.indices.size();
                const BufferCreateInfo indexBufferCreateInfo {
                    .allocator = mContext->allocator(),
                    .size = indicesSize,
                    .bufferUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
                };
                meshBuffer.indexBuffer = Buffer(indexBufferCreateInfo);
                const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                    .data = const_cast<uint32_t*>(meshData.indices.data()),
                    .size = indicesSize,
                    .allocator = mContext->allocator(),
                    .buffer = meshBuffer.indexBuffer.get(),
                    .device = mContext->device(),
                    .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                    .queue = mContext->queue(QueueFamilyType::GRAPHICS)
                };
                Buffer::copy(copyDataToGPUBufferInfo);
            }
            meshBuffer.firstInstance = instanceModels.size();
            for (auto instance: meshData.instances) instanceModels.push_back(instance.model);
        }
        {
            const size_t instancesSize = sizeof(glm::mat4) * instanceModels.size();
            const BufferCreateInfo instanceBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = instancesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mInstanceBuffer = Buffer(instanceBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<glm::mat4*>(instanceModels.data()),
                .size = instancesSize,
                .allocator = mContext->allocator(),
                .buffer = mInstanceBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
    }
}
