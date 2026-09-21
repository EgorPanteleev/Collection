//
// Created by igor on 4/13/26.
//

#include "ComputePipelines.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    ComputePipelines::ComputePipelines(const ComputePipelinesCreateInfo &info): mDevice(info.device) {
        std::vector<VkComputePipelineCreateInfo> pipelineInfos;
        for (size_t i = 0; i < info.stages.size(); ++i) {
            VkComputePipelineCreateInfo pipelineInfo{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = info.stages[i],
                .layout = info.layouts[i]
            };
            pipelineInfos.push_back(pipelineInfo);
        }

        mVec.resize(pipelineInfos.size());
        if (vkCreateComputePipelines(mDevice, VK_NULL_HANDLE,
                static_cast<uint32_t>(pipelineInfos.size()), pipelineInfos.data(),
                nullptr, mVec.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipelines!");
        }
    }

    void ComputePipelines::destroy() {
        if (mDevice == VK_NULL_HANDLE or mVec.empty()) return;
        for (auto& pipeline: mVec) {
            vkDestroyPipeline(mDevice, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        mVec.clear();
        mDevice = VK_NULL_HANDLE;
    }
}
