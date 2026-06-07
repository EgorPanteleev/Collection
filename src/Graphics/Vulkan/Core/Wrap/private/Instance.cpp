//
// Created by igor on 4/11/26.
//

#include "Instance.hpp"

#include <cstring>
#include <unordered_set>
#include <GL/glew.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace crv::graphics::vulkan {
    Instance::Instance(const InstanceCreateInfo& info) {
        if (info.enableValidationLayers && !checkValidationLayerSupport(info.validationLayers)) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }
        constexpr VkApplicationInfo appInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_4
        };

        const auto& extensions = getRequiredExtensions(info.enableValidationLayers, info.enableRT);
        VkInstanceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        const/*expr*/ VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = createDebugMessengerCreateInfo();
        if (info.enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(info.validationLayers.size());
            createInfo.ppEnabledLayerNames = info.validationLayers.data();
            createInfo.pNext = &debugCreateInfo;
        }

        if (vkCreateInstance(&createInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance!");
        }
        INFO << "VkInstance created!";
        checkGlfwRequiredInstanceExtensions(info.enableValidationLayers, info.enableRT);
    }

    void Instance::destroy() {
        if (mHandle == VK_NULL_HANDLE) return;
        vkDestroyInstance(mHandle, nullptr);
    }

    bool Instance::checkValidationLayerSupport(const std::vector<const char*>& validationLayers) {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) return false;
        }
        return true;
    }

    std::vector<const char *> Instance::getRequiredExtensions(bool enableValidationLayers, bool enableRT) {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (enableRT) extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        return extensions;
    }

    static constexpr VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(const VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                              VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                              void *pUserData) {

        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            ERROR << "ERROR: " << pCallbackData->pMessage;
        } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            WARNING << "WARNING: " << pCallbackData->pMessage;
        } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            INFO << "INFO: " << pCallbackData->pMessage;
        } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
            DEBUG << "VERBOSE: " << pCallbackData->pMessage;
        }

        return VK_FALSE;
    }

    const VkDebugUtilsMessengerCreateInfoEXT Instance::createDebugMessengerCreateInfo() {
        constexpr VkDebugUtilsMessengerCreateInfoEXT createInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback
        };
        return createInfo;
    }

    void Instance::checkGlfwRequiredInstanceExtensions(bool enableValidationLayers, bool enableRT) {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        INFO << "Available extensions:";
        std::unordered_set<std::string> available;
        for (const auto &extension : extensions) {
            INFO << "\t" << extension.extensionName;
            available.insert(extension.extensionName);
        }

        INFO << "Required extensions:";
        const auto requiredExtensions = getRequiredExtensions(enableValidationLayers, enableRT);
        for (const auto &required : requiredExtensions) {
            INFO << "\t" << required;
            if (available.find(required) != available.end()) continue;
            throw std::runtime_error("Missing required glfw extension");
        }
    }
}