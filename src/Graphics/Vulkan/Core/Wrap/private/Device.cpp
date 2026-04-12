//
// Created by igor on 4/12/26.
//

#include "Device.hpp"
#include "Message.hpp"

#include <set>
#include <stdexcept>

namespace crv::graphics::vulkan {
    Device::Device(const DeviceCreateInfo& info) {
        create(info);
    }

    void Device::create(const DeviceCreateInfo& info) {
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set uniqueQueueFamilies = {info.familyIndices.get(QueueFamilyType::COMPUTE ).value(),
                                        info.familyIndices.get(QueueFamilyType::GRAPHICS).value(),
                                        info.familyIndices.get(QueueFamilyType::PRESENT ).value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{
            .sampleRateShading = VK_TRUE,
            .samplerAnisotropy = VK_TRUE
        };

        VkPhysicalDeviceVulkan12Features vulkan12Features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
                .descriptorBindingPartiallyBound = VK_TRUE,
                .descriptorBindingVariableDescriptorCount = VK_TRUE,
                .runtimeDescriptorArray = VK_TRUE,
                .separateDepthStencilLayouts = VK_TRUE,
        };

        VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeature{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
                .pNext = &vulkan12Features,
                .dynamicRendering = VK_TRUE,
        };

        VkDeviceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &dynamicRenderingFeature,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(info.deviceExtensions.size()),
            .ppEnabledExtensionNames = info.deviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures,
        };
        if (info.enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(info.validationLayers.size());
            createInfo.ppEnabledLayerNames = info.validationLayers.data();
        }

        if (vkCreateDevice(info.physicalDevice, &createInfo, nullptr, &mHandle) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device!");

        INFO << "Logical device created!";
    }

    void Device::destroy() {
        if (mHandle == VK_NULL_HANDLE) return;
        vkDestroyDevice(mHandle, nullptr);
    }

    VkQueue Device::getQueue(const uint32_t index) const {
        VkQueue queue;
        vkGetDeviceQueue(mHandle, index, 0, &queue);
        return queue;
    }
}
