//
// Created by igor on 4/13/26.
//

#include "PipelineLayout.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    PipelineLayout::PipelineLayout(const PipelineLayoutCreateInfo &info): mDevice(info.device) {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = info.layouts.data(),
            .pushConstantRangeCount = static_cast<uint32_t>(info.ranges.size()),
            .pPushConstantRanges = info.ranges.data()
    };

        if (vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }
        INFO << "Created pipeline layout!";
    }

    void PipelineLayout::destroy() {
        if (mDevice == VK_NULL_HANDLE) return;
        vkDestroyPipelineLayout(mDevice, mHandle, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
}
