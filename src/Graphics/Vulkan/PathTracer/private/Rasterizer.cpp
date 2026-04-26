//
// Created by igor on 4/26/26.
//

#include "Rasterizer.hpp"

namespace crv::graphics::vulkan {
    Rasterizer::Rasterizer(const RasterizerCreateInfo& info): mFramesInFlight(info.framesInFlight), mColorFormat(info.colorFormat),
    mContext(info.context), mTextures(info.textures) {
        createColorBuffer(info);
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
            AlignedMVP MVP {
                .model = glm::rotate(glm::mat4(1.0f), glm::radians(-0.0f), glm::vec3(1, 0, 0)),
                .view = info.camera->viewMatrix(),
                .proj = info.camera->projectionMatrix()
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

        std::vector descriptorWrites{
            writeDescriptorSet0
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
        //TODO create GBUFFER, next to..
    }

    void Rasterizer::createColorBuffer(const RasterizerCreateInfo& info) {
        const ImageCreateInfo createInfo {
            .device = mContext->device(),
            .allocator = mContext->allocator(),
            .format = mColorFormat,
            .extent = info.extent,
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        mColorBuffer = Image(createInfo);
    }

    void Rasterizer::createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding MVPBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = nullptr
        };

        std::vector bindings = {MVPBinding};

        std::vector<VkDescriptorBindingFlags> bindingFlags = {
                0,
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
        const std::vector variableCounts(mFramesInFlight, static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)); //mTexturesSize
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
            .colorFormat = mColorFormat,
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
    }
}
