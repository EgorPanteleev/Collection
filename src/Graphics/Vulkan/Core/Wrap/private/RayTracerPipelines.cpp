//
// Created by igor on 6/7/26.
//

#include "RayTracerPipelines.hpp"

namespace crv::graphics::vulkan {
    RayTracerPipelines::RayTracerPipelines(const RayTracerPipelinesCreateInfo &info): mDevice(info.device) {
        const VkRayTracingPipelineCreateInfoKHR pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .flags = 0,
            .stageCount = static_cast<uint32_t>(info.stages.size()),
            .pStages = info.stages.data(),
            .groupCount = static_cast<uint32_t>(info.groups.size()),
            .pGroups = info.groups.data(),
            .maxPipelineRayRecursionDepth = 1,
            .layout = info.layout,
        };
        mVec.resize(1);
        LOAD_VK_FN(mDevice, vkCreateRayTracingPipelinesKHR);
        vkCreateRayTracingPipelinesKHR(
            mDevice, VK_NULL_HANDLE, VK_NULL_HANDLE,
            1, &pipelineInfo, nullptr, mVec.data());
    }

    void RayTracerPipelines::destroy() {
        if (mDevice == VK_NULL_HANDLE or mVec.empty()) return;
        for (auto& pipeline: mVec) {
            vkDestroyPipeline(mDevice, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        mVec.clear();
        mDevice = VK_NULL_HANDLE;
    }
}