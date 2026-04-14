//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_IMAGEVIEW_HPP
#define COLLECTION_IMAGEVIEW_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct ImageViewCreateInfo {
        VkDevice device                   = VK_NULL_HANDLE;
        VkImage  image                    = VK_NULL_HANDLE;
        VkImageViewType    viewType       = VK_IMAGE_VIEW_TYPE_2D;
        VkFormat           format         = VK_FORMAT_MAX_ENUM;
        VkImageAspectFlags aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        uint32_t           mipLevels      = 1;
        uint32_t           baseMipLevel   = 0;
        uint32_t           baseArrayLayer = 0;
        uint32_t           layerCount     = 1;
    };

    class ImageView: public DefaultWrapper<VkImageView> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit ImageView(const ImageViewCreateInfo& info);
        explicit ImageView(ImageView&&) = default;
        ImageView& operator=(ImageView&&) = default;
        ~ImageView() override { ImageView::destroy(); }
        void destroy() override;
        static VkImageView create(const ImageViewCreateInfo& info);
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_IMAGEVIEW_HPP