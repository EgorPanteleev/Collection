//
// Created by igor on 4/13/26.
//

#include "CommandPool.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    CommandPool::CommandPool(const CommandPoolCreateInfo &info): mDevice(info.device) {
        const VkCommandPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = info.flags,
            .queueFamilyIndex = info.queueFamilyIndex,
        };
        if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void CommandPool::destroy() {
        if (mDevice == VK_NULL_HANDLE) return;
        vkDestroyCommandPool(mDevice, mHandle, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
}
;