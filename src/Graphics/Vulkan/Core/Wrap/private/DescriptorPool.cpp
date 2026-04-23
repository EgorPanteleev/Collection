//
// Created by igor on 4/13/26.
//

#include "DescriptorPool.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    DescriptorPool::DescriptorPool(const DescriptorPoolCreateInfo &info): mDevice(info.device) {
        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = info.flags,
            .maxSets = info.maxSets,
            .poolSizeCount = static_cast<uint32_t>(info.poolSizes.size()),
            .pPoolSizes = info.poolSizes.data()
    };
        if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
        INFO << "Created descriptor pool!";
    }

    void DescriptorPool::destroy() {
        if (mDevice == VK_NULL_HANDLE) return;
        vkDestroyDescriptorPool(mDevice, mHandle, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
}
