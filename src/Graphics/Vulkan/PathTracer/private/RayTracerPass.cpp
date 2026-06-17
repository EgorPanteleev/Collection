//
// Created by igor on 6/7/26.
//

#include "RayTracerPass.hpp"

namespace crv::graphics::vulkan {
    RayTracerPass::RayTracerPass(const RayTracerPassCreateInfo& info):
    mFramesInFlight(info.framesInFlight), mContext(info.context), mTLAS(info.tlas),
    mBLASBuffer(info.BLASBuffer), mInstanceBuffer(info.instanceBuffer), mEmissiveInstanceBuffer(info.emissiveInstanceBuffer),
    mTextures(info.textures), mMaterialBuffer(info.materialBuffer), mOutputView(info.outView),
    mOutputInstanceIdView(info.outInstanceIdView) {
        createBuffers();
        createDescriptorManager();
        createShaders();
        createPipelineLayout();
        createPipelines();
        createSBT();
    }

    void RayTracerPass::update(const RayTracerPassUpdateInfo& info) {
        Buffer& cameraBuffer = mCameraBuffers[info.currentFrame];
        const CameraGPU camera {
            .invView = glm::inverse(info.camera->viewMatrix()),
            .invProj = glm::inverse(info.camera->projectionMatrix())
        };
        copyDataToBuffer(mContext, QueueFamilyType::GRAPHICS, &camera, sizeof(CameraGPU), cameraBuffer);

        Buffer& directLightBuffer = mDirectLightBuffers[info.currentFrame];
        copyDataToBuffer(mContext, QueueFamilyType::GRAPHICS, &info.directLight, sizeof(DirectLight), directLightBuffer);
    }

    void RayTracerPass::record(const RayTracerPassRecordInfo& info) {
        vkCmdBindPipeline(info.commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, mPipelines[0]);
        vkCmdBindDescriptorSets(info.commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            mPipelineLayout.get(), 0, 1, &mDescriptorManager.set(0),
            0, nullptr);

        vkCmdPushConstants(info.commandBuffer, mPipelineLayout.get(),
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            0, sizeof(PushConstants), &info.constants);

        LOAD_VK_FN(mContext->device(), vkCmdTraceRaysKHR);
        vkCmdTraceRaysKHR(info.commandBuffer,
            &mRaygenRegion,
            &mMissRegion,
            &mHitRegion,
            &mCallRegion,
            info.width, info.height, 1);
    }

    void RayTracerPass::createDescriptorManager() {
        auto textureCount = static_cast<uint32_t>(mTextures->size());
        mDescriptorManager.add(BindingType::AS           , VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        mDescriptorManager.add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR     );
        mDescriptorManager.add(BindingType::STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR     );
        mDescriptorManager.add(BindingType::UBO          , VK_SHADER_STAGE_RAYGEN_BIT_KHR     );
        mDescriptorManager.add(BindingType::UBO          , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        mDescriptorManager.add(BindingType::SSBO         , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        mDescriptorManager.add(BindingType::SSBO         , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        mDescriptorManager.add(BindingType::SSBO         , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        mDescriptorManager.add(BindingType::SSBO         , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        mDescriptorManager.add(BindingType::TEXTURE      , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, textureCount);

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
            .variableCount = static_cast<uint32_t>(mTextures->size())
        };
        mDescriptorManager.build(buildInfo);

        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager.bind(i, ASResource(mTLAS->get()));
            mDescriptorManager.bind(i, ImageResource(mOutputView, VK_IMAGE_LAYOUT_GENERAL));
            mDescriptorManager.bind(i, ImageResource(mOutputInstanceIdView, VK_IMAGE_LAYOUT_GENERAL));
            mDescriptorManager.bind(i, BufferResource(mCameraBuffers[i]));
            mDescriptorManager.bind(i, BufferResource(mDirectLightBuffers[i]));
            mDescriptorManager.bind(i, BufferResource(*mBLASBuffer));
            mDescriptorManager.bind(i, BufferResource(*mInstanceBuffer));
            mDescriptorManager.bind(i, BufferResource(*mEmissiveInstanceBuffer));
            mDescriptorManager.bind(i, BufferResource(*mMaterialBuffer));
            mDescriptorManager.bind(i, ImageArrayResource(*mTextures));
        }
        mDescriptorManager.update();
    }

    void RayTracerPass::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
        };
        createInfo.fileName = COMPILED_SHADERS_DIR"/pathtracer.slang.spv";
        mRaygenShader = ShaderModule(createInfo);
        mMissShader = ShaderModule(createInfo);
        mShadowMissShader = ShaderModule(createInfo);
        mHitShader = ShaderModule(createInfo);
    }

    void RayTracerPass::createPipelineLayout() {
        VkPushConstantRange pushRange {
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
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

    void RayTracerPass::createPipelines() {
        const std::vector<VkPipelineShaderStageCreateInfo> stages {
            { //raygen
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .flags = 0,
                .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                .module = mRaygenShader.get(),
                .pName = "rgenMain",
            },
            { //miss
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .flags = 0,
                .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                .module = mMissShader.get(),
                .pName = "missMain",
            },
            { //shadow miss
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .flags = 0,
                .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                .module = mShadowMissShader.get(),
                .pName = "shadowMissMain",
            },
            { //closest hit
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .flags = 0,
                .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                .module = mHitShader.get(),
                .pName = "chitMain",
            },
        };
        const std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups = {
            { //raygen
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader = 0,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
            },
            { //miss
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader = 1,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
            },
            { //shadow miss
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader = 2,
                .closestHitShader = VK_SHADER_UNUSED_KHR,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
            },
            { //closest hit
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                .generalShader = VK_SHADER_UNUSED_KHR,
                .closestHitShader = 3,
                .anyHitShader = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
            }
        };

        const RayTracerPipelinesCreateInfo createInfo {
            .device = mContext->device(),
            .stages = stages,
            .groups = groups,
            .layout = mPipelineLayout.get()
        };
        mPipelines = RayTracerPipelines(createInfo);
    }

    void RayTracerPass::createSBT() {
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        };
        VkPhysicalDeviceProperties2 props2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &props
        };
        vkGetPhysicalDeviceProperties2(mContext->physicalDevice(), &props2);
        uint32_t groupCount = 4;
        uint32_t handleSize = props.shaderGroupHandleSize;
        std::vector<uint8_t> handles(groupCount * handleSize);
        LOAD_VK_FN(mContext->device(), vkGetRayTracingShaderGroupHandlesKHR);
        vkGetRayTracingShaderGroupHandlesKHR(mContext->device(), mPipelines[0],
            0, groupCount, handles.size(), handles.data());

        constexpr auto alignUp = [](const uint32_t v, const uint32_t a){ return (v + a - 1) & ~(a - 1); };
        uint32_t stride = alignUp(props.shaderGroupHandleSize, props.shaderGroupHandleAlignment);
        uint32_t baseAlign     = props.shaderGroupBaseAlignment;
        uint32_t raygenOffset     = 0;
        uint32_t missOffset       = alignUp(raygenOffset     + stride, baseAlign);
        uint32_t shadowMissOffset = alignUp(missOffset       + stride, baseAlign);
        uint32_t hitOffset        = alignUp(shadowMissOffset + stride, baseAlign);
        uint32_t sbtSize          = hitOffset + stride;

        const BufferCreateInfo sbtCreateInfo {
            .allocator = mContext->allocator(),
            .size = sbtSize,
            .bufferUsage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO,
            .minAlignment = baseAlign
        };
        mSBTBuffer = Buffer(sbtCreateInfo);

        const BufferCreateInfo stagingBufferCreateInfo {
            .allocator = mContext->allocator(),
            .size = sbtSize,
            .bufferUsage = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
            .allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST
        };
        Buffer stagingBuffer = Buffer(stagingBufferCreateInfo);
        auto mapped = static_cast<uint8_t*>(stagingBuffer.map());
        memcpy(mapped + raygenOffset    , handles.data() + 0 * handleSize, handleSize);
        memcpy(mapped + missOffset      , handles.data() + 1 * handleSize, handleSize);
        memcpy(mapped + shadowMissOffset, handles.data() + 2 * handleSize, handleSize);
        memcpy(mapped + hitOffset       , handles.data() + 3 * handleSize, handleSize);
        stagingBuffer.unmap();

        CopyBufferToBufferInfo copyInfo {
            .srcBuffer = stagingBuffer.get(),
            .dstBuffer = mSBTBuffer.get(),
            .size = stagingBuffer.size(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);

        VkBufferDeviceAddressInfo addrInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = mSBTBuffer.get()
        };
        VkDeviceAddress sbtAddr = vkGetBufferDeviceAddress(mContext->device(), &addrInfo);

        mRaygenRegion     = {sbtAddr + raygenOffset    , stride, stride};
        mMissRegion       = {sbtAddr + missOffset      , stride, stride * 2};
        mHitRegion        = {sbtAddr + hitOffset       , stride, stride};
        mCallRegion       = {};
    }

    void RayTracerPass::createBuffers() {
        mCameraBuffers.resize(mFramesInFlight);
        mDirectLightBuffers.resize(mFramesInFlight);
        UBOData uboData{};
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            uboData.add<CameraGPU>(mCameraBuffers[i]);
            uboData.add<DirectLight>(mDirectLightBuffers[i]);
        }
        uboData.createAll(mContext);
    }
}
