//
// Created by igor on 4/14/26.
//

#include "Fence.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    Fence::Fence(const FenceCreateInfo &info): mDevice(info.device) {
        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        if (vkCreateFence(mDevice, &fenceInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence!");
        }
    }

    void Fence::destroy() {
        if (mDevice == VK_NULL_HANDLE or mHandle == VK_NULL_HANDLE) return;
        vkDestroyFence(mDevice, mHandle, nullptr);
    }
}