//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const WindowCreateInfo& windowCreateInfo) {
        createContext(windowCreateInfo);
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets();
        createPipelineLayout();
        createShaders();
        createComputePipelines();
    }

    void PathTracer::createContext(const WindowCreateInfo& windowCreateInfo) {
        const ContextCreateInfo createInfo {
            .windowCreateInfo = windowCreateInfo,
            .validationLayers = { "VK_LAYER_KHRONOS_validation" },
            .deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                     VK_KHR_MAINTENANCE_1_EXTENSION_NAME },
            .enableValidationLayers = mDebug
        };
        mContext = Context(createInfo);
    }

    void PathTracer::createDescriptorSetLayout() {
        const VkDescriptorSetLayoutBinding binding0 {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        const VkDescriptorSetLayoutBinding binding1 {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };

        std::vector bindings{binding0,
                             binding1};
        std::vector<VkDescriptorBindingFlags> bindingFlags{0,
                                                           0};
        const DescriptorSetLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .bindings = bindings,
            .bindingFlags = bindingFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }
    void PathTracer::createDescriptorPool() {
        std::vector<VkDescriptorPoolSize> poolSizes {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext.device(),
            .poolSizes = poolSizes,
            .maxSets = 1
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void PathTracer::createDescriptorSets() {
        std::vector<uint32_t> variableCounts{0, 0};
        DescriptorSetsCreateInfo createInfo {
            .device = mContext.device(),
            .layouts = getDescriptorLayouts(),
            .pool = mDescriptorPool.get(),
            .variableCounts =variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);
    }

    void PathTracer::createPipelineLayout() {
        PipelineLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .layouts = getDescriptorLayouts()
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void PathTracer::createShaders() {
        const ShaderModuleCreateInfo createInfo {
            .device = mContext.device(),
            .fileName = COMPILED_SHADERS_DIR"/shader1.comp.spv"
        };
        mShader = ShaderModule(createInfo);
    }

    void PathTracer::createComputePipelines() {
        std::vector<VkPipelineShaderStageCreateInfo> stages {
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
        ComputePipelinesCreateInfo createInfo {
            .device = mContext.device(),
            .stages = stages,
            .layouts = getPipelineLayouts(),
        };
        mComputePipelines = ComputePipelines(createInfo);
    }

    std::vector<VkDescriptorSetLayout> PathTracer::getDescriptorLayouts() const {
        std::vector<VkDescriptorSetLayout> layouts;
        layouts.push_back(mDescriptorSetLayout.get());
        return layouts;
    }

    std::vector<VkPipelineLayout> PathTracer::getPipelineLayouts() const {
        std::vector<VkPipelineLayout> layouts;
        layouts.push_back(mPipelineLayout.get());
        return layouts;
    }
}

/* Plan
 * 1) Creating context
 *   1.1) window, surface
 *   1.2) instance
 *   1.3) devices
 *   1.4) queues
 *   1.5) allocator
 *
 * 2) Buffers - resources
 *   2.1) BVH
 *   2.2) Materials
 *   2.3) Lights
 *
 * 3) Bind resources
 *   3.1) Descriptor Set Layout
 *   3.2) Descriptor Set
 *
 * 4) Bind descriptor sets
 *   4.1) Write shaders, compile it
 *   4.2) Compute Pipeline Layout
 *   4.3) Compute Pipeline
 *
 * 5) Command buffers, recording
 *   5.1) Record command buffer
 *   5.2) Submit command buffer
 *
 * 6) Present the results
*/