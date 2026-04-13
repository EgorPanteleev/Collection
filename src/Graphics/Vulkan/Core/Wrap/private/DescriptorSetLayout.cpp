//
// Created by igor on 4/13/26.
//

#include "DescriptorSetLayout.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    DescriptorSetLayout::DescriptorSetLayout(const DescriptorSetLayoutCreateInfo &info): mDevice(info.device) {
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .bindingCount = static_cast<uint32_t>(info.bindingFlags.size()),
                .pBindingFlags = info.bindingFlags.data()
        };
        const VkDescriptorSetLayoutCreateInfo layoutInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &bindingFlagsInfo,
                .bindingCount = static_cast<uint32_t>(info.bindings.size()),
                .pBindings = info.bindings.data()
        };
        if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
        INFO << "Created descriptor set layout!";
    }

    void DescriptorSetLayout::destroy() {
        if (mDevice == VK_NULL_HANDLE) return;
        vkDestroyDescriptorSetLayout(mDevice, mHandle, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
}
