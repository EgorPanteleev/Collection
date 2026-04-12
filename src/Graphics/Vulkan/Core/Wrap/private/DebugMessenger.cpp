//
// Created by igor on 4/12/26.
//

#include "DebugMessenger.hpp"
#include "Instance.hpp"
#include "Message.hpp"

static VkResult createDebugUtilsMessengerEXT(VkInstance instance,
                              const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                              const VkAllocationCallbacks* pAllocator,
                              VkDebugUtilsMessengerEXT* pDebugMessenger) {
    const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>
    (vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (!func) return VK_ERROR_EXTENSION_NOT_PRESENT;
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
}

namespace crv::graphics::vulkan {
    DebugMessenger::DebugMessenger(const DebugMessengerCreateInfo &info): mInstance(info.instance) {
        const auto createInfo = Instance::createDebugMessengerCreateInfo();
        if (createDebugUtilsMessengerEXT(mInstance, &createInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug messenger!");
        }
        INFO << "Created debug messenger!";
    }

    void DebugMessenger::destroy() {
        if (mHandle == VK_NULL_HANDLE) return;
        const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>
        (vkGetInstanceProcAddr(mInstance, "vkDestroyDebugUtilsMessengerEXT"));
        if (func == nullptr) return;
        func(mInstance, mHandle, nullptr);
    }
}
