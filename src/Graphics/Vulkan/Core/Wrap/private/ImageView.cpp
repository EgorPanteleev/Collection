//
// Created by igor on 4/14/26.
//

#include "ImageView.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    ImageView::ImageView(const ImageViewCreateInfo &info): mDevice(info.device) {
        mHandle = create(info);
    }

    void ImageView::destroy() {
        if (mDevice == VK_NULL_HANDLE or mHandle == VK_NULL_HANDLE) return;
        vkDestroyImageView(mDevice, mHandle, nullptr);
        mDevice = VK_NULL_HANDLE;
    }

    VkImageView ImageView::create(const ImageViewCreateInfo &info) {
        VkImageView view;
        const VkImageViewCreateInfo imageViewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = info.image,
            .viewType = info.viewType,
            .format = info.format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
                .aspectMask = info.aspectMask,
                .baseMipLevel = info.baseMipLevel,
                .levelCount = info.mipLevels,
                .baseArrayLayer = info.baseArrayLayer,
                .layerCount = info.layerCount,
            },
        };
        const VkResult result = vkCreateImageView(info.device, &imageViewInfo, nullptr, &view);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view!");
        }
        return view;
    }
}
