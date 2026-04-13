//
// Created by igor on 4/13/26.
//

#include "ComputePipelines.hpp"

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

        vkCreateComputePipelines(mDevice,VK_NULL_HANDLE,
            1, pipelineInfos.data(), nullptr, mVec.data());
    }

    void ComputePipelines::destroy() {
        for (auto& pipeline: mVec) {
            vkDestroyPipeline(mDevice, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        mVec.clear();
        mDevice = VK_NULL_HANDLE;
    }
}
