//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Camera.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const PathTracerCreateInfo& info): mFramesInFlight(info.framesInFlight),
    mTexturesSize(info.materials.size() * static_cast<uint32_t>(cm::Texture::UNKNOWN)), mContext(info.context) {
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets();
        createPipelineLayout();
        createShaders();
        createComputePipelines();
        createBuffers(info);
        createTextures(info);
    }
    
    void PathTracer::update(const PathTracerUpdateInfo& info) {
        Buffer& cameraBuffer = mCameraBuffers[info.currentFrame];
        {
            AlignedCamera camera {
                .position = Vec4(info.camera->position(), 1),
                .forward = Vec4(info.camera->forward(), 1),
                .right = Vec4(info.camera->right(), 1),
                .up = Vec4(info.camera->up(), 1),
                .FOV = info.camera->FOV(),
                .aspectRatio = info.camera->aspectRatio(),
                .nearPlane = info.camera->nearPlane(),
                .farPlane = info.camera->farPlane()
            };
            const BufferCreateInfo cameraBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = sizeof(AlignedCamera),
                .bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            cameraBuffer = Buffer(cameraBufferCreateInfo);
            CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = &camera,
                .size = sizeof(AlignedCamera),
                .allocator = mContext->allocator(),
                .buffer = cameraBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext->queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        Buffer& directLightBuffer = mDirectLightBuffers[info.currentFrame];
        {
            const BufferCreateInfo directLightBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = sizeof(AlignedDirectLight),
                .bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            directLightBuffer = Buffer(directLightBufferCreateInfo);
            CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<void*>(static_cast<const void*>(&info.directLight)),
                .size = sizeof(AlignedDirectLight),
                .allocator = mContext->allocator(),
                .buffer = directLightBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext->queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        VkDescriptorBufferInfo cameraBufferInfo {
            .buffer = cameraBuffer.get(),
            .offset = 0,
            .range = cameraBuffer.size()
        };
        VkDescriptorBufferInfo directLightBufferInfo {
            .buffer = directLightBuffer.get(),
            .offset = 0,
            .range = directLightBuffer.size()
        };
        VkDescriptorBufferInfo triangleBufferInfo {
            .buffer = mTriangleBuffer.get(),
            .offset = 0,
            .range = mTriangleBuffer.size()
        };
        VkDescriptorBufferInfo triangleExtraBufferInfo {
            .buffer = mTriangleExtraBuffer.get(),
            .offset = 0,
            .range = mTriangleExtraBuffer.size()
        };
        VkDescriptorBufferInfo nodeBufferInfo {
            .buffer = mNodeBuffer.get(),
            .offset = 0,
            .range = mNodeBuffer.size()
        };
        VkDescriptorBufferInfo materialIndexBufferInfo {
            .buffer = mMaterialIndexBuffer.get(),
            .offset = 0,
            .range = mMaterialIndexBuffer.size()
        };
        std::vector<VkDescriptorImageInfo> textureInfos;
        for (size_t i = 0; i < mTextures.size(); i++) {
            auto& texturesByType = mTextures[i];
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

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext->device(), mContext->familyIndex(QueueFamilyType::COMPUTE).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const ImageTransitInfo imageTransitInfo {
            .commandBuffer = commandBuffer,
            .image = info.presentImage,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        };
        Image::transit(imageTransitInfo);

        endCommandBuffer(commandPool, commandBuffers, mContext->queue(QueueFamilyType::COMPUTE));

        VkDescriptorImageInfo imageInfo {
            .sampler = VK_NULL_HANDLE,
            .imageView = info.presentImageView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL
        };

        VkWriteDescriptorSet writeDescriptorSet0 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &cameraBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet1 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &triangleBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet2 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &triangleExtraBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet3 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &nodeBufferInfo,
            .pTexelBufferView = nullptr
        };
        VkWriteDescriptorSet writeDescriptorSet4 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 4,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &materialIndexBufferInfo,
            .pTexelBufferView = nullptr
        };
        VkWriteDescriptorSet writeDescriptorSet5 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 5,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &directLightBufferInfo,
            .pTexelBufferView = nullptr
        };
        VkWriteDescriptorSet writeDescriptorSet6 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 6,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfo,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr
        };
        VkWriteDescriptorSet writeDescriptorSet7 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 7,
            .dstArrayElement = 0,
            .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = textureInfos.data()
        };
        std::vector descriptorWrites{
            writeDescriptorSet0, writeDescriptorSet1, writeDescriptorSet2,
            writeDescriptorSet3, writeDescriptorSet4, writeDescriptorSet5,
            writeDescriptorSet6, writeDescriptorSet7
        };
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            descriptorsWrites.push_back(descriptorWrites);
        }

        DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
    }

    void PathTracer::record(const PathTracerRecordInfo& info) {
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mComputePipelines[0]);
        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineLayout.get(),
                        0, 1, &mDescriptorSets[info.currentFrame], 0, nullptr);

        const uint32_t width  = info.extent.width;
        const uint32_t height = info.extent.height;
        const PushConstants pc(width, height, info.frameCount);
        vkCmdPushConstants(info.commandBuffer, mPipelineLayout.get(), VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);

        const uint32_t groupX = 1 + (width - 1 ) / 8;
        const uint32_t groupY = 1 + (height - 1) / 8;
        vkCmdDispatch(info.commandBuffer, groupX, groupY, 1);
    }

    std::vector<VkDescriptorSetLayout> PathTracer::getDescriptorLayouts() const {
        std::vector<VkDescriptorSetLayout> layouts;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            layouts.push_back(mDescriptorSetLayout.get());
        }
        return layouts;
    }

    std::vector<VkPipelineLayout> PathTracer::getPipelineLayouts() const {
        std::vector<VkPipelineLayout> layouts;
        layouts.push_back(mPipelineLayout.get());
        return layouts;
    }

    void PathTracer::createDescriptorSetLayout() {
        constexpr VkDescriptorSetLayoutBinding binding0 {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding1 {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding2 {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding3 {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding4 {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding5 {
            .binding = 5,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding6 {
            .binding = 6,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        VkDescriptorSetLayoutBinding binding7 {
            .binding = 7,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = mTexturesSize,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };

        const std::vector bindings{binding0, binding1, binding2, binding3,
                                   binding4, binding5, binding6, binding7};
        const std::vector<VkDescriptorBindingFlags> bindingFlags{0, 0, 0, 0, 0, 0, 0,
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
        };
        const DescriptorSetLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .bindings = bindings,
            .bindingFlags = bindingFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }

    void PathTracer::createDescriptorPool() {
        const std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER , mFramesInFlight * mTexturesSize},
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext->device(),
            .poolSizes = poolSizes,
            .maxSets = mFramesInFlight
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void PathTracer::createDescriptorSets() {
        const std::vector variableCounts(mFramesInFlight, mTexturesSize);
        const DescriptorSetsCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = getDescriptorLayouts(),
            .pool = mDescriptorPool.get(),
            .variableCounts =variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);
    }

    void PathTracer::createPipelineLayout() {
        VkPushConstantRange pushRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = getDescriptorLayouts(),
            .ranges = {pushRange}
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void PathTracer::createShaders() {
        const ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
            .fileName = COMPILED_SHADERS_DIR"/pathTracer.comp.spv"
        };
        mShader = ShaderModule(createInfo);
    }

    void PathTracer::createComputePipelines() {
        const std::vector<VkPipelineShaderStageCreateInfo> stages {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mShader.get(),
                .pName = "main",
                .pSpecializationInfo = nullptr
            },
        };
        const ComputePipelinesCreateInfo createInfo {
            .device = mContext->device(),
            .stages = stages,
            .layouts = getPipelineLayouts(),
        };
        mComputePipelines = ComputePipelines(createInfo);
    }

    void PathTracer::createBuffers(const PathTracerCreateInfo& info) {
        {
            const uint32_t trianglesSize = sizeof(AlignedTriangle) * info.triangles.size();
            const BufferCreateInfo triangleBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = trianglesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mTriangleBuffer = Buffer(triangleBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<AlignedTriangle*>(info.triangles.data()),
                .size = trianglesSize,
                .allocator = mContext->allocator(),
                .buffer = mTriangleBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext->queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        {
            const uint32_t triangleExtrasSize = sizeof(AlignedTriangleExtra) * info.triangleExtras.size();
            const BufferCreateInfo vertexBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = triangleExtrasSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mTriangleExtraBuffer = Buffer(vertexBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<AlignedTriangleExtra*>(info.triangleExtras.data()),
                .size = triangleExtrasSize,
                .allocator = mContext->allocator(),
                .buffer = mTriangleExtraBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext->queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        {
            const uint32_t nodesSize = sizeof(AlignedNode) * info.nodes.size();
            const BufferCreateInfo nodeBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = nodesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mNodeBuffer = Buffer(nodeBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<AlignedNode*>(info.nodes.data()),
                .size = nodesSize,
                .allocator = mContext->allocator(),
                .buffer = mNodeBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext->queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        {
            const uint32_t indicesSize = sizeof(uint32_t) * info.materialIndices.size();
            const BufferCreateInfo materialIndexBufferCreateInfo {
                .allocator = mContext->allocator(),
                .size = indicesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mMaterialIndexBuffer = Buffer(materialIndexBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<uint32_t*>(info.materialIndices.data()),
                .size = indicesSize,
                .allocator = mContext->allocator(),
                .buffer = mMaterialIndexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext->queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        mCameraBuffers.resize(mFramesInFlight);
        mDirectLightBuffers.resize(mFramesInFlight);
    }

    void PathTracer::createTextures(const PathTracerCreateInfo& info) {
        mTextures.resize(info.materials.size());
        for (size_t i = 0; i < info.materials.size(); ++i) {
            const cm::Material& material = info.materials[i];
            TexturesByType& texturesByType = mTextures[i];
            for (int texType = 0; texType < static_cast<int>(cm::Texture::UNKNOWN); ++texType) {
                const cm::Texture& texture = material.mTextures[texType];
                TextureCreateInfo textureCreateInfo {
                    .device = mContext->device(),
                    .physicalDevice = mContext->physicalDevice(),
                    .allocator = mContext->allocator(),
                    .queue = mContext->queue(QueueFamilyType::COMPUTE),
                    .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::COMPUTE).value(),
                    .dataByLevel = texture.mDataByLevel,
                    .texFormat = texture.mFormat,
                    .mipLevels = 1,
                    .arrayLayers = 1,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .memoryUsage = VMA_MEMORY_USAGE_AUTO
                };
                texturesByType[texType] = Texture(textureCreateInfo);
            }
        }
    }
}