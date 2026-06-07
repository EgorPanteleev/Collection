//
// Created by igor on 4/12/26.
//

#include "Allocator.hpp"

namespace crv::graphics::vulkan {
    Allocator::Allocator(const AllocatorCreateInfo &info) {
        const VmaAllocatorCreateInfo allocatorInfo = {
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = info.physicalDevice,
            .device = info.device,
            .instance = info.instance
        };
        vmaCreateAllocator(&allocatorInfo, &mHandle);
    }

    void Allocator::destroy() {
        if (mHandle == VK_NULL_HANDLE) return;
        vmaDestroyAllocator(mHandle);
    }
}
