//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_IMAGE_HPP
#define COLLECTION_IMAGE_HPP

#include "DefaultWrapper.hpp"
#include "vk_mem_alloc.h"
#include "Allocation.hpp"

namespace crv::graphics::vulkan {
    struct ImageCreateInfo {
        VkDevice              device    = VK_NULL_HANDLE;
        VmaAllocator          allocator = VK_NULL_HANDLE;
        VkImageCreateFlags    flags  = 0;
        VkFormat              format = VK_FORMAT_MAX_ENUM;
        VkExtent3D            extent{};
        uint32_t              mipLevels   = 1;
        uint32_t              arrayLayers = 1;
        VkSampleCountFlagBits samples     = VK_SAMPLE_COUNT_1_BIT;
        VkImageTiling         tiling      = VK_IMAGE_TILING_OPTIMAL;
        VkImageUsageFlags     imageUsage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        VmaMemoryUsage        memoryUsage = VMA_MEMORY_USAGE_AUTO;
    };

    struct ImageTransitInfo {
        VkCommandBuffer      commandBuffer  = VK_NULL_HANDLE;
        VkImage              image          = VK_NULL_HANDLE;
        VkAccessFlags        srcAccessMask  = VK_ACCESS_NONE;
        VkAccessFlags        dstAccessMask  = VK_ACCESS_NONE;
        VkImageLayout        oldLayout      = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout        newLayout      = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageAspectFlags   aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        uint32_t             baseMipLevel   = 0;
        uint32_t             levelCount     = 1;
        uint32_t             baseArrayLayer = 0;
        uint32_t             layerCount     = 1;
        VkPipelineStageFlags srcStage       = VK_PIPELINE_STAGE_NONE;
        VkPipelineStageFlags dstStage       = VK_PIPELINE_STAGE_NONE;
    };

    struct CopyBufferToImageInfo {
        VkCommandBuffer       commandBuffer  = VK_NULL_HANDLE;
        VkBuffer              buffer         = VK_NULL_HANDLE;
        VkImage               image          = VK_NULL_HANDLE;
        VkExtent2D            extent{};
        uint32_t              mipLevel       = 1;
        uint32_t              baseArrayLayer = 0;
        uint32_t              layerCount     = 1;
    };

    class Image: public DefaultWrapper<VkImage> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Image(const ImageCreateInfo& info);
        Image(Image&&) = default;
        Image& operator=(Image&&) = default;
        ~Image() override { Image::destroy(); }
        void destroy() override;
        [[nodiscard]] VmaAllocation allocation() const { return mAllocation.get(); }
        static void transit(const ImageTransitInfo& info);
        static void copy(const CopyBufferToImageInfo& info);
    protected:
        void createImage(const ImageCreateInfo& info);

        VkDevice mDevice = VK_NULL_HANDLE;
        VmaAllocator mAllocator = VK_NULL_HANDLE;
        Allocation mAllocation{};
    };
}

#endif //COLLECTION_IMAGE_HPP