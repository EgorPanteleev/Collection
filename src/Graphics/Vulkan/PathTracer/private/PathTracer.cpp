//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Camera.hpp"
#include "Types.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const PathTracerCreateInfo& info): mFramesInFlight(info.framesInFlight),
    mInstanceCount(info.instances.size()), mTextures(info.textures), mContext(info.context) {
        createBuffers(info);
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets(info);
        createPipelineLayout();
        createShaders();
        createComputePipelines();
    }
    
    void PathTracer::update(const PathTracerUpdateInfo& info) {
        Buffer& cameraBuffer = mCameraBuffers[info.currentFrame];
        AlignedCamera camera {
            .pos = Vec4(info.camera->position(), 1),
            .invViewProj = glm::inverse(info.camera->projectionMatrix() * info.camera->viewMatrix())
        };
        copyDataToBuffer(mContext, QueueFamilyType::COMPUTE, &camera, sizeof(AlignedCamera), cameraBuffer);

        Buffer& directLightBuffer = mDirectLightBuffers[info.currentFrame];
        copyDataToBuffer(mContext, QueueFamilyType::COMPUTE, const_cast<AlignedDirectLight*>(&info.directLight), sizeof(AlignedDirectLight), directLightBuffer);
    }

    void PathTracer::record(const PathTracerRecordInfo& info) {
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mComputePipelines[0]);
        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineLayout.get(),
                        0, 1, &mDescriptorSets[info.currentFrame], 0, nullptr);

        const PushConstants pc(info.frameCount, info.spp, info.minDepth, info.maxDepth);
        vkCmdPushConstants(info.commandBuffer, mPipelineLayout.get(), VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);

        auto [width, height]  = info.extent;
        const uint32_t groupX = 1 + (width  - 1) / 16;
        const uint32_t groupY = 1 + (height - 1) / 16;
        vkCmdDispatch(info.commandBuffer, groupX, groupY, 1);
    }

    std::vector<VkDescriptorSetLayout> PathTracer::getDescriptorLayouts() const {
        std::vector<VkDescriptorSetLayout> layouts;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            layouts.push_back(mDescriptorSetLayout.get());
        }
        return layouts;
    }

    void PathTracer::createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding texturesBinding {
            .binding = 11,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN),
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };
        const std::vector bindings{
            getLayoutBinding(0 , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(1 , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(2 , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(3 , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(4 , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(5 , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(6 , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER        , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(7 , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE         , VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(8 , VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(9 , VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT),
            getLayoutBinding(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT),
            texturesBinding
        };
        const std::vector<VkDescriptorBindingFlags> bindingFlags{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER        , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE         , mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mFramesInFlight
                * static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)},
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext->device(),
            .poolSizes = poolSizes,
            .maxSets = mFramesInFlight
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void PathTracer::createDescriptorSets(const PathTracerCreateInfo& info) {
        const std::vector variableCounts(mFramesInFlight, static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN));
        const DescriptorSetsCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = getDescriptorLayouts(),
            .pool = mDescriptorPool.get(),
            .variableCounts =variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);

        std::vector<VkDescriptorImageInfo> textureInfos;
        for (size_t i = 0; i < mTextures->size(); i++) {
            auto& texturesByType = (*mTextures)[i];
            for (int j = 0; j < cm::Texture::Type::UNKNOWN; ++j) {
                auto& texture = texturesByType[j];
                if (texture.view() == VK_NULL_HANDLE) continue;
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
            .image = info.outImage,
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

        VkWriteDescriptorSet texturesWriteDescriptorSet {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 11,
            .dstArrayElement = 0,
            .descriptorCount = static_cast<uint32_t>(textureInfos.size()),
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = textureInfos.data()
        };

        std::vector<VkDescriptorBufferInfo> bufferInfos;
        std::vector<VkDescriptorImageInfo> imageInfos;
        bufferInfos.reserve(60);
        imageInfos.reserve(60);
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            GBuffer& gBuffer = (*info.gBuffers)[i];
            std::vector descriptorWrites{
                getUBODescriptorWrite (mCameraBuffers[i]   , 0, bufferInfos),
                getSSBODescriptorWrite(mTriangleBuffer     , 1, bufferInfos),
                getSSBODescriptorWrite(mTriangleExtraBuffer, 2, bufferInfos),
                getSSBODescriptorWrite(mNodeBuffer         , 3, bufferInfos),
                getSSBODescriptorWrite(mTLASNodeBuffer     , 4, bufferInfos),
                getSSBODescriptorWrite(mInstanceBuffer     , 5, bufferInfos),
                getUBODescriptorWrite (mDirectLightBuffers[i], 6, bufferInfos),
                getStorageImageDescriptorWrite(info.outImageView, VK_IMAGE_LAYOUT_GENERAL, 7, imageInfos),
                getSamplerImageDescriptorWrite(gBuffer.sampler.get(), gBuffer.colorView.get(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 8, imageInfos),
                getSamplerImageDescriptorWrite(gBuffer.sampler.get(), gBuffer.depthView.get(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 9, imageInfos),
                getSamplerImageDescriptorWrite(gBuffer.sampler.get(), gBuffer.normalView.get(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 10, imageInfos),
                texturesWriteDescriptorSet
            };
            descriptorsWrites.push_back(descriptorWrites);
        }

        const DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
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
            .layouts = {mPipelineLayout.get()},
        };
        mComputePipelines = ComputePipelines(createInfo);
    }

    void PathTracer::createBuffers(const PathTracerCreateInfo& info) {
        SSBOData ssboData{};
        ssboData.add(info.triangles     , mTriangleBuffer     );
        ssboData.add(info.triangleExtras, mTriangleExtraBuffer);
        ssboData.add(info.nodes         , mNodeBuffer         );
        ssboData.add(info.TLASNodes     , mTLASNodeBuffer     );
        ssboData.add(info.instances     , mInstanceBuffer     );
        ssboData.createAll(mContext, QueueFamilyType::COMPUTE);

        mCameraBuffers.resize(mFramesInFlight);
        mDirectLightBuffers.resize(mFramesInFlight);
        UBOData uboData{};
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            uboData.add<AlignedCamera     >(mCameraBuffers[i]     );
            uboData.add<AlignedDirectLight>(mDirectLightBuffers[i]);
        }
        uboData.createAll(mContext, QueueFamilyType::COMPUTE);
    }
}