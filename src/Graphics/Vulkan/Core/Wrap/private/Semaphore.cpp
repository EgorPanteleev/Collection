//
// Created by igor on 4/14/26.
//

#include "Semaphore.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    Semaphore::Semaphore(const SemaphoreCreateInfo &info): mDevice(info.device) {
        const VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        };
        if (vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create semaphore!");
        }
    }

    void Semaphore::destroy() {
        if (mDevice == VK_NULL_HANDLE or mHandle == VK_NULL_HANDLE) return;
        vkDestroySemaphore(mDevice, mHandle, nullptr);
    }
}
