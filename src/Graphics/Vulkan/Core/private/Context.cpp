//
// Created by auser on 4/2/25.
//

#include "Context.hpp"
#include "Message.hpp"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <cstring>
#include <stdexcept>
#include <iostream>
#include <unordered_set>
#include <set>

#include "Device.hpp"

namespace crv::graphics::vulkan {
    Context::Context(const ContextCreateInfo &createInfo) {
        mValidationLayers = createInfo.validationLayers;
        mDeviceExtensions = createInfo.deviceExtensions;
        mEnableValidationLayers = createInfo.enableValidationLayers;

        mWindow = Window(createInfo.windowCreateInfo);
        const InstanceCreateInfo instanceCreateInfo{
            .validationLayers = mValidationLayers,
            .enableValidationLayers = mEnableValidationLayers
        };
        mInstance = Instance(instanceCreateInfo);
        const SurfaceCreateInfo surfaceCreateInfo{
            .instance = mInstance.get(),
            .window = mWindow.glfwWindow()
        };
        mSurface = Surface(surfaceCreateInfo);
        const DebugMessengerCreateInfo messengerCreateInfo{
            .instance = mInstance.get()
        };
        if (mEnableValidationLayers)
            mDebugMessenger = DebugMessenger(messengerCreateInfo);

        pickPhysicalDevice();
        mFamilyIndices = getQueueFamilies(mPhysicalDevice);

        const DeviceCreateInfo deviceCreateInfo {
            .familyIndices = mFamilyIndices,
            .physicalDevice = mPhysicalDevice,
            .validationLayers = mValidationLayers,
            .deviceExtensions = mDeviceExtensions,
            .enableValidationLayers = mEnableValidationLayers
        };
        mDevice = Device(deviceCreateInfo);
        mComputeQueue  = mDevice.getQueue(mFamilyIndices.get(QueueFamilyType::COMPUTE ).value());
        mGraphicsQueue = mDevice.getQueue(mFamilyIndices.get(QueueFamilyType::GRAPHICS).value());
        mPresentQueue  = mDevice.getQueue(mFamilyIndices.get(QueueFamilyType::PRESENT ).value());

        const AllocatorCreateInfo allocatorCreateInfo{
            .physicalDevice = mPhysicalDevice,
            .device = mDevice.get(),
            .instance = mInstance.get()
        };
        mAllocator = Allocator(allocatorCreateInfo);
    }

    VkQueue Context::queue(const QueueFamilyType type) const {
        switch (type) {
            case QueueFamilyType::COMPUTE:
                return mComputeQueue;
            case QueueFamilyType::GRAPHICS:
                return mGraphicsQueue;
            case QueueFamilyType::PRESENT:
                return mPresentQueue;
            default: {
                ERROR << "There is no queue with given type!";
                return {};
            }
        }
    }

    void Context::pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(mInstance.get(), &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("failed to find GPUs with Vulkan support!");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(mInstance.get(), &deviceCount, devices.data());
        INFO << "Available devices:";
        for (const auto& device: devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            INFO << "    " << deviceProperties.deviceName;
        }

        for (const auto& device: devices) {
            if ( !isDeviceSuitable(device)) continue;
            mPhysicalDevice = device;
            break;
        }

        if ( mPhysicalDevice == VK_NULL_HANDLE )
            throw std::runtime_error("Failed to find a suitable GPU!");

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(mPhysicalDevice, &deviceProperties);
        INFO << "Picked physical device:\n    " << deviceProperties.deviceName;
    }

    bool Context::isDeviceSuitable(VkPhysicalDevice device) const {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            return false;
        }

        const auto familyIndices = getQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapChainSupported = extensionsSupported and checkSwapChainSupport(device);

        VkPhysicalDeviceFeatures supportedFeatures{};
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
        return familyIndices.isComplete() && extensionsSupported &&
               swapChainSupported && supportedFeatures.samplerAnisotropy;
    }

    QueueFamilyIndices Context::getQueueFamilies(VkPhysicalDevice device) const {
        QueueFamilyIndices indices;
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
                indices.set(QueueFamilyType::COMPUTE, i);
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.set(QueueFamilyType::GRAPHICS, i);
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, mSurface.get(), &presentSupport);
            if (presentSupport) indices.set(QueueFamilyType::PRESENT, i);
            if ( indices.isComplete() ) break;
            ++i;
        }
        return indices;
    }

    bool Context::checkDeviceExtensionSupport(VkPhysicalDevice device) const {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(mDeviceExtensions.begin(), mDeviceExtensions.end());
        for (const auto& extension: availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    bool Context::checkSwapChainSupport(VkPhysicalDevice device) const {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats{};
        std::vector<VkPresentModeKHR> presentModes{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, mSurface.get(), &capabilities);
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, mSurface.get(), &formatCount, nullptr);

        if (formatCount != 0) {
            formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, mSurface.get(), &formatCount, formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, mSurface.get(), &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, mSurface.get(), &presentModeCount, presentModes.data());
        }
        return !formats.empty() && !presentModes.empty();
    }
}
