//
// Created by igor on 4/13/26.
//

#include "CommandBuffers.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    CommandBuffers::CommandBuffers(const CommandBuffersCreateInfo &info): mDevice(info.device), mCommandPool(info.commandPool) {
        const VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = mCommandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = info.bufferCount
        };
        mVec.resize(info.bufferCount);
        if (vkAllocateCommandBuffers(mDevice, &allocInfo, mVec.data() ) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }
    }

    void CommandBuffers::destroy() {
        if (mDevice == VK_NULL_HANDLE or mVec.empty()) return;
        vkFreeCommandBuffers(mDevice, mCommandPool, mVec.size(), mVec.data());
        mDevice = VK_NULL_HANDLE;
        mCommandPool = VK_NULL_HANDLE;
    }
}
