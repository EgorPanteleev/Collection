//
// Created by igor on 6/7/26.
//

#include "RayTracerPass.hpp"
#include "TypesGPU.hpp"

namespace crv::graphics::vulkan {
    RayTracerPass::RayTracerPass(const RayTracerPassCreateInfo& info):
    mContext(info.context), mOutView(info.outView), mFramesInFlight(info.framesInFlight) {
        createDescriptorManager();
        createShaders();
        createPipelineLayout();
        createPipelines();
        createSBT();
    }

    void RayTracerPass::createDescriptorManager() {
        constexpr BindingDescription storageImageBinding {
            .type   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        mDescriptorManager.add(storageImageBinding );

        const DescriptorBuildInfo buildInfo {
            .context = mContext,
            .count = mFramesInFlight,
            .variableCount = 0//static_cast<uint32_t>(mTextures->size() * cm::Texture::UNKNOWN)
        };
        mDescriptorManager.build(buildInfo);

        for (int i = 0; i < mFramesInFlight; ++i) {
            mDescriptorManager.add(i, ImageResource(mOutView, VK_IMAGE_LAYOUT_GENERAL));
        }
        mDescriptorManager.update();
    }

    void RayTracerPass::createShaders() {
        ShaderModuleCreateInfo createInfo {
            .device = mContext->device(),
        };
        createInfo.fileName = COMPILED_SHADERS_DIR"/raygen.rgen.spv";
        mRaygenShader = ShaderModule(createInfo);
        createInfo.fileName = COMPILED_SHADERS_DIR"/miss.rmiss.spv";
        mMissShader = ShaderModule(createInfo);
        createInfo.fileName = COMPILED_SHADERS_DIR"/closesthit.rchit.spv";
        mHitShader = ShaderModule(createInfo);
    }

    void RayTracerPass::createPipelineLayout() {
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext->device(),
            .layouts = mDescriptorManager.layouts(mFramesInFlight),
            .ranges = {}
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
                .pName = "main",
            },
            { //miss
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .flags = 0,
                .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                .module = mMissShader.get(),
                .pName = "main",
            },
            { //closest hit
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .flags = 0,
                .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                .module = mHitShader.get(),
                .pName = "main",
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
            { //closest hit
                .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                .generalShader = VK_SHADER_UNUSED_KHR,
                .closestHitShader = 2,
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
        uint32_t groupCount = 3;
        uint32_t handleSize = props.shaderGroupHandleSize;
        std::vector<uint8_t> handles(groupCount * handleSize);
        LOAD_VK_FN(mContext->device(), vkGetRayTracingShaderGroupHandlesKHR);
        vkGetRayTracingShaderGroupHandlesKHR(mContext->device(), mPipelines[0],
            0, groupCount, handles.size(), handles.data());

        constexpr auto alignUp = [](const uint32_t v, const uint32_t a){ return (v + a - 1) & ~(a - 1); };
        uint32_t stride = alignUp(props.shaderGroupHandleSize, props.shaderGroupHandleAlignment);
        uint32_t baseAlign     = props.shaderGroupBaseAlignment;
        uint32_t raygenOffset = 0;
        uint32_t missOffset   = alignUp(raygenOffset + stride, baseAlign);
        uint32_t hitOffset    = alignUp(missOffset   + stride, baseAlign);
        uint32_t sbtSize      = hitOffset + stride;

        const BufferCreateInfo sbtCreateInfo {
            .allocator = mContext->allocator(),
            .size = sbtSize,
            .bufferUsage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
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
        memcpy(mapped + raygenOffset, handles.data() + 0 * handleSize, handleSize);
        memcpy(mapped + missOffset,   handles.data() + 1 * handleSize, handleSize);
        memcpy(mapped + hitOffset,    handles.data() + 2 * handleSize, handleSize);
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

        mRaygenRegion = {sbtAddr + raygenOffset, stride, stride};
        mMissRegion   = {sbtAddr + missOffset,   stride, stride};
        mHitRegion    = {sbtAddr + hitOffset,    stride, stride};
        mCallRegion   = {};
    }
}
