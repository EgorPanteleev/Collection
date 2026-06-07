//
// Created by igor on 4/12/26.
//

#ifndef COLLECTION_DEVICE_HPP
#define COLLECTION_DEVICE_HPP

#include <vulkan/vulkan_core.h>
#include "DefaultWrapper.hpp"
#include "QueueFamily.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    struct DeviceCreateInfo {
        QueueFamilyIndices familyIndices{};
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        std::vector<const char *> validationLayers{};
        std::vector<const char *> deviceExtensions{};
        bool enableValidationLayers = false;
        bool enableRT               = false;
    };

    class Device: public DefaultWrapper<VkDevice> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Device(const DeviceCreateInfo& info);
        Device& operator=(Device&&) = default;
        ~Device() override { Device::destroy(); }
        void destroy() override;
        [[nodiscard]] VkQueue getQueue(uint32_t index) const;
    protected:
        void create(const DeviceCreateInfo& info);
    };
}

#endif //COLLECTION_DEVICE_HPP