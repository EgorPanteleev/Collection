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
        if (mHandle == VK_NULL_HANDLE or mDevice == VK_NULL_HANDLE) return;
        vmaDestroyImage(mAllocator, mHandle, mAllocation.get());
        mDevice = VK_NULL_HANDLE;
        mAllocation.destroy();
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
                                               &mHandle, &mAllocation.get(), nullptr);
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

    void Image::transit(const std::vector<ImageTransitInfo2>& infos) {
        if (infos.empty()) return;
        std::vector<VkImageMemoryBarrier2> barriers;
        barriers.reserve(infos.size());
        for (const auto& info: infos) {
            const VkImageMemoryBarrier2 barrier {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = info.srcStage,
                .srcAccessMask = info.srcAccessMask,
                .dstStageMask = info.dstStage,
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
            barriers.push_back(barrier);
        }

        const VkDependencyInfo depInfo{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data()
        };
        vkCmdPipelineBarrier2(infos[0].commandBuffer, &depInfo);
    }

    void Image::transit(const ImageTransitInfo2& info) {
        transit(std::vector{info});
    }

    void Image::inverseTransit(const ImageTransitInfo& info) {
        const VkImageMemoryBarrier barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = info.dstAccessMask,
            .dstAccessMask = info.srcAccessMask,
            .oldLayout = info.newLayout,
            .newLayout = info.oldLayout,
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

        vkCmdPipelineBarrier(info.commandBuffer, info.dstStage, info.srcStage,
            0,0, nullptr,0,
            nullptr,1, &barrier);
    }

    void Image::inverseTransit(const std::vector<ImageTransitInfo2>& infos) {
        if (infos.empty()) return;
        std::vector<VkImageMemoryBarrier2> barriers;
        barriers.reserve(infos.size());
        for (const auto& info: infos) {
            const VkImageMemoryBarrier2 barrier {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = info.dstStage,
                .srcAccessMask = info.dstAccessMask,
                .dstStageMask = info.srcStage,
                .dstAccessMask = info.srcAccessMask,
                .oldLayout = info.newLayout,
                .newLayout = info.oldLayout,
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
            barriers.push_back(barrier);
        }

        const VkDependencyInfo depInfo{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers = barriers.data()
        };
        vkCmdPipelineBarrier2(infos[0].commandBuffer, &depInfo);
    }

    void Image::inverseTransit(const ImageTransitInfo2& info) {
        inverseTransit(std::vector{info});
    }

    VkImageMemoryBarrier Image::barrier(const ImageBarrierInfo& info) {
        return {
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
    }

    VkImageMemoryBarrier Image::inverseBarrier(const ImageBarrierInfo& info) {
        return {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = info.dstAccessMask,
            .dstAccessMask = info.srcAccessMask,
            .oldLayout = info.newLayout,
            .newLayout = info.oldLayout,
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
    }

    void Image::copy(const CopyBufferToImageInfo &info) {
        const VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = info.mipLevel,
                .baseArrayLayer = info.baseArrayLayer,
                .layerCount = info.layerCount
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {info.extent.width, info.extent.height, 1}
        };
        vkCmdCopyBufferToImage(
                info.commandBuffer, info.buffer, info.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    }

    void Image::pipelineBarrier(const ImagePipelineBarrierInfo& info) {
        vkCmdPipelineBarrier(info.commandBuffer, info.srcStage, info.dstStage,
    0,0, nullptr,0,
    nullptr,info.barriers.size(), info.barriers.data());
    }

    void Image::inversePipelineBarrier(const ImagePipelineBarrierInfo& info) {
        vkCmdPipelineBarrier(info.commandBuffer, info.dstStage, info.srcStage,
    0,0, nullptr,0,
    nullptr,info.barriers.size(), info.barriers.data());
    }
}
