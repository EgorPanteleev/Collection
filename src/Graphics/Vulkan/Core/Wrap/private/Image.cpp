//
// Created by igor on 4/14/26.
//

#include "Image.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    Image::Image(const ImageCreateInfo &info): mDevice(info.device), mAllocator(info.allocator) {
        createImage(info);
    }

    void Image::destroy() {
        if (mDevice == VK_NULL_HANDLE) return;
        vmaDestroyImage(mAllocator, mHandle, mAllocation);
        mDevice = VK_NULL_HANDLE;
        mAllocator = VK_NULL_HANDLE;
    }

    void Image::createImage(const ImageCreateInfo& info) {
        const VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = info.flags,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = info.format,
            .extent = info.extent,
            .mipLevels = info.mipLevels,
            .arrayLayers = info.arrayLayers,
            .samples = info.samples,
            .tiling = info.tiling,
            .usage = info.imageUsage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        const VmaAllocationCreateInfo allocInfo{
            .usage = info.memoryUsage,
            .preferredFlags = 0,
        };

        const VkResult result = vmaCreateImage(info.allocator, &imageInfo, &allocInfo,
                                               &mHandle, &mAllocation, nullptr);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image!");
        }
    }

    void Image::transit(const ImageTransitInfo& info) {
        const VkImageMemoryBarrier barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = info.srcAccessMask,
            .dstAccessMask = info.dstAccessMask,
            .oldLayout = info.oldLayout,
            .newLayout = info.newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = info.image,
            .subresourceRange = {
                .aspectMask = info.aspectMask,
                .baseMipLevel = info.baseMipLevel,
                .levelCount = info.levelCount,
                .baseArrayLayer = info.baseArrayLayer,
                .layerCount = info.layerCount
            }
        };

        vkCmdPipelineBarrier(info.commandBuffer, info.srcStage, info.dstStage,
            0,0, nullptr,0,
            nullptr,1, &barrier);
    }
}
