//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_SWAPCHAIN_HPP
#define COLLECTION_SWAPCHAIN_HPP

#include "DefaultWrapper.hpp"
#include "QueueFamily.hpp"

namespace crv::graphics::vulkan {
    struct SwapchainCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        uint32_t windowWidth = 0;
        uint32_t windowHeight = 0;
        QueueFamilyIndices familyIndices{};
    };

    struct SwapchainSupport {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats{};
        std::vector<VkPresentModeKHR> presentModes{};
    };

    class Swapchain: public DefaultWrapper<VkSwapchainKHR> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Swapchain(const SwapchainCreateInfo& info);
        Swapchain& operator=(Swapchain&&) = default;
        ~Swapchain() override { Swapchain::destroy(); }
        void destroy() override;
        [[nodiscard]] VkFormat format() const { return mFormat; }
        [[nodiscard]] VkExtent2D extent() const { return mExtent; }
        static SwapchainSupport getSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
        static uint32_t getImageCount(VkSurfaceCapabilitiesKHR capabilities);
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
        VkFormat mFormat = VK_FORMAT_MAX_ENUM;
        VkExtent2D mExtent{};
    };
}

#endif //COLLECTION_SWAPCHAIN_HPP