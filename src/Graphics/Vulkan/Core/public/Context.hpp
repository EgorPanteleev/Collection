//
// Created by auser on 4/2/25.
//

#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include <vk_mem_alloc.h>
#include <vector>
#include <array>
#include <optional>

#include "Window.hpp"
#include "Instance.hpp"
#include "Surface.hpp"
#include "DebugMessenger.hpp"
#include "QueueFamily.hpp"
#include "Device.hpp"
#include "Allocator.hpp"

namespace crv::graphics::vulkan {
    struct ContextCreateInfo {
        WindowCreateInfo windowCreateInfo{};
        std::vector<const char*> validationLayers{};
        std::vector<const char*> deviceExtensions{};
        bool enableValidationLayers = false;
    };

    class Context {
    public:
        Context() = default;
        explicit Context(const ContextCreateInfo &createInfo);
        [[nodiscard]] VkInstance instance() const { return mInstance.get(); }
        [[nodiscard]] const Window& window() const { return mWindow; }
        [[nodiscard]] Window& window() { return mWindow; }
        [[nodiscard]] GLFWwindow* glfwWindow() const { return mWindow.glfwWindow(); }
        [[nodiscard]] VkSurfaceKHR surface() const { return mSurface.get(); }
        [[nodiscard]] VkPhysicalDevice physicalDevice() const { return mPhysicalDevice; }
        [[nodiscard]] const QueueFamilyIndices& familyIndices() const { return mFamilyIndices; }
        [[nodiscard]] VkDevice device() const { return mDevice.get(); }
        [[nodiscard]] VmaAllocator allocator() const { return mAllocator.get(); }
    protected:
        void pickPhysicalDevice();
        bool isDeviceSuitable(VkPhysicalDevice device) const;
        QueueFamilyIndices getQueueFamilies(VkPhysicalDevice device) const;
        bool checkDeviceExtensionSupport(VkPhysicalDevice device) const;
        bool checkSwapChainSupport(VkPhysicalDevice device) const;

        std::vector<const char *> mValidationLayers{};
        std::vector<const char *> mDeviceExtensions{};
        bool mEnableValidationLayers = false;

        Window mWindow{};
        Instance mInstance{};
        Surface mSurface{};
        DebugMessenger mDebugMessenger{};
        VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
        QueueFamilyIndices mFamilyIndices{};
        Device mDevice{};
        Allocator mAllocator{};
        VkQueue mComputeQueue = VK_NULL_HANDLE;
        VkQueue mGraphicsQueue = VK_NULL_HANDLE;
        VkQueue mPresentQueue = VK_NULL_HANDLE;
    };

}


#endif //VULKAN_CONTEXT_H
