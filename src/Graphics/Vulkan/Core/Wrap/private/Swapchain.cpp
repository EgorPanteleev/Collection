//
// Created by igor on 4/14/26.
//

#include "Swapchain.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                INFO << "Present mode: Mailbox";
                return availablePresentMode;
            }
        }
        INFO << "Present mode: V-Sync";
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, const uint32_t width, const uint32_t height) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            VkExtent2D actualExtent = {
                width,
                height
            };
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }

    Swapchain::Swapchain(const SwapchainCreateInfo &info): mDevice(info.device) {
        const auto [capabilities, formats, presentModes] = getSupport(info.physicalDevice, info.surface);
        const auto [format, colorSpace] = chooseSwapSurfaceFormat(formats);
        const VkPresentModeKHR presentMode = chooseSwapPresentMode(presentModes);
        mExtent = chooseSwapExtent(capabilities, info.windowWidth, info.windowHeight);
        mFormat = format;

        const uint32_t imageCount = getImageCount(capabilities);
        VkSwapchainCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = info.surface,
            .minImageCount = imageCount,
            .imageFormat = format,
            .imageColorSpace = colorSpace,
            .imageExtent = mExtent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE
        };

        uint32_t familyIndices[] = {info.familyIndices.get(QueueFamilyType::COMPUTE).value(),
                                    info.familyIndices.get(QueueFamilyType::PRESENT).value()};
        if (familyIndices[0] != familyIndices[1]) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = familyIndices;
        }

        if (vkCreateSwapchainKHR(mDevice, &createInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swap chain!");
        }
        INFO << "Created swapchain!";
    }

    void Swapchain::destroy() {
        if (mDevice == VK_NULL_HANDLE) return;
        vkDestroySwapchainKHR(mDevice, mHandle, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
    
    VkResult Swapchain::acquireNextImage(const SwapchainAcquireInfo& info) const {
        vkWaitForFences(mDevice, 1, &info.fence, VK_TRUE, UINT64_MAX);
        const VkResult result = vkAcquireNextImageKHR(mDevice, mHandle, UINT64_MAX,
                                                 info.imageAvailableSemaphore,
                                                 VK_NULL_HANDLE, info.imageIndex);
        return result;
    }

    SwapchainSupport Swapchain::getSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
        SwapchainSupport support;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &support.capabilities);
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            support.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, support.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            support.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, support.presentModes.data());
        }
        return support;
    }

    uint32_t Swapchain::getImageCount(const VkSurfaceCapabilitiesKHR capabilities) {
        uint32_t imageCount = capabilities.minImageCount + 1;

        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }
        return imageCount;
    }
}
