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
    mInstanceCount(info.instances.size()), mTextures(info.textures), mContext(info.context),
    mOutImageView(info.outImageView) {
        createBuffers(info);
        createDescriptorManager(info);
        createPipelineLayout();
        createShaders();
        createComputePipelines();
    }
    
    void PathTracer::update(const PathTracerUpdateInfo& info) {
        Buffer& cameraBuffer = mCameraBuffers[info.currentFrame];
        CameraGPU camera {
            .pos = glm::vec4(info.camera->position(), 1),
            .invViewProj = glm::inverse(info.camera->projectionMatrix() * info.camera->viewMatrix())
        };
        copyDataToBuffer(mContext, QueueFamilyType::COMPUTE, &camera, sizeof(CameraGPU), cameraBuffer);

        Buffer& directLightBuffer = mDirectLightBuffers[info.currentFrame];
        copyDataToBuffer(mContext, QueueFamilyType::COMPUTE, &info.directLight, sizeof(DirectLightGPU), directLightBuffer);
    }

    void PathTracer::record(const PathTracerRecordInfo& info) {
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mComputePipelines[0]);
        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineLayout.get(),
                        0, 1, &mDescriptorManager.set(info.currentFrame), 0, nullptr);

        vkCmdPushConstants(info.commandBuffer, mPipelineLayout.get(), VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &info.constants);

        auto [width, height]  = info.extent;
        const uint32_t groupX = 1 + (width  - 1) / 16;
        const uint32_t groupY = 1 + (height - 1) / 16;
        vkCmdDispatch(info.commandBuffer, groupX, groupY, 1);
    }

    void PathTracer::updateTLAS(const std::vector<NodeGPU>& nodes, const std::vector<MeshInstance>& instances, const std::vector<GBuffer>& gBuffers) {
        std::vector<TracerInstanceGPU> tracerInstances;
        for (const auto& instance: instances) tracerInstances.emplace_back(instance);
        copyDataToBuffer(mContext, QueueFamilyType::COMPUTE, tracerInstances.data(),
            tracerInstances.size() * sizeof(TracerInstanceGPU), mInstanceBuffer);

        uint32_t newSize = nodes.size() * sizeof(NodeGPU);
        if (mTLASNodeBuffer.size() != newSize) {
            createSSBO(mContext->allocator(), newSize, mTLASNodeBuffer);
            mDescriptorManager.update(4, BufferResource(mTLASNodeBuffer));
            mDescriptorManager.update();
        }
        copyDataToBuffer(mContext, QueueFamilyType::COMPUTE, nodes.data(),
            newSize, mTLASNodeBuffer);
    }

    void PathTracer::createDescriptorManager(const PathTracerCreateInfo& info) {
        //layout
        constexpr BindingDescription UBOBinding {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        constexpr BindingDescription SSBOBinding {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        constexpr BindingDescription storageImageBinding {
            .type   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        constexpr BindingDescription samplerImageBinding {
            .type   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        const BindingDescription texturesBinding {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
            .count = static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)
        };

        mDescriptorManager.add(UBOBinding );
        mDescriptorManager.add(SSBOBinding);
        mDescriptorManager.add(SSBOBinding);
        mDescriptorManager.add(SSBOBinding);
        mDescriptorManager.add(SSBOBinding);
        mDescriptorManager.add(SSBOBinding);
        mDescriptorManager.add(UBOBinding );
        mDescriptorManager.add(storageImageBinding );
        mDescriptorManager.add(samplerImageBinding );
        mDescriptorManager.add(samplerImageBinding );
        mDescriptorManager.add(samplerImageBinding );
        mDescriptorManager.add(texturesBinding );

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
            .variableCount = static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)
        };
        mDescriptorManager.build(buildInfo);

        //resources
        for (int i = 0; i < mFramesInFlight; ++i) {
            GBuffer& gBuffer = (*info.gBuffers)[i];
            mDescriptorManager.add(i, BufferResource(mCameraBuffers[i]   ));
            mDescriptorManager.add(i, BufferResource(mTriangleBuffer     ));
            mDescriptorManager.add(i, BufferResource(mTriangleExtraBuffer));
            mDescriptorManager.add(i, BufferResource(mNodeBuffer         ));
            mDescriptorManager.add(i, BufferResource(mTLASNodeBuffer     ));
            mDescriptorManager.add(i, BufferResource(mInstanceBuffer     ));
            mDescriptorManager.add(i, BufferResource(mDirectLightBuffers[i] ));

            VkSampler sampler = gBuffer.sampler.get();
            mDescriptorManager.add(i, ImageResource(VK_NULL_HANDLE, mOutImageView, VK_IMAGE_LAYOUT_GENERAL));
            mDescriptorManager.add(i, ImageResource(sampler, gBuffer.colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
            mDescriptorManager.add(i, ImageResource(sampler, gBuffer.depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
            mDescriptorManager.add(i, ImageResource(sampler, gBuffer.normalView,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
            mDescriptorManager.add(i, ImageResource(*mTextures));
        }
        mDescriptorManager.update();

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(), mContext->familyIndex(QueueFamilyType::COMPUTE).value());
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
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::COMPUTE));
    }

    void PathTracer::createPipelineLayout() {
        VkPushConstantRange pushRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
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
        std::vector<TracerInstanceGPU> instances;
        for (const auto& instance: info.instances) instances.emplace_back(instance);
        ssboData.add(info.triangles     , mTriangleBuffer     );
        ssboData.add(info.triangleExtras, mTriangleExtraBuffer);
        ssboData.add(info.nodes         , mNodeBuffer         );
        ssboData.add(info.tlasNodes     , mTLASNodeBuffer     );
        ssboData.add(instances          , mInstanceBuffer     );
        ssboData.createAll(mContext, QueueFamilyType::COMPUTE);

        mCameraBuffers.resize(mFramesInFlight);
        mDirectLightBuffers.resize(mFramesInFlight);
        UBOData uboData{};
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            uboData.add<CameraGPU     >(mCameraBuffers[i]     );
            uboData.add<DirectLightGPU>(mDirectLightBuffers[i]);
        }
        uboData.createAll(mContext);
    }
}