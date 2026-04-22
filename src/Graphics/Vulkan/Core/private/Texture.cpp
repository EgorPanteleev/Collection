//
// Created by igor on 4/19/26.
//
#include "Texture.hpp"
#include "Message.hpp"
#include "CoreUtils.hpp"
#include "Buffer.hpp"

namespace crv::graphics::vulkan {
    Texture::Texture(const TextureCreateInfo& info) {
        create(info);
        load(info);
    }

    static VkFormat toVkFormat( cm::Texture::Format format ) {
        switch(format) {
            case cm::Texture::R8G8B8A8_SRGB:
                return VK_FORMAT_R8G8B8A8_SRGB;
            case cm::Texture::R8G8B8A8_UNORM:
                return VK_FORMAT_R8G8B8A8_UNORM;
            case cm::Texture::BC1_UNORM:
                return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            case cm::Texture::BC3_UNORM:
                return VK_FORMAT_BC3_UNORM_BLOCK;
            case cm::Texture::BC5_UNORM:
                return VK_FORMAT_BC5_UNORM_BLOCK;
            default:
                throw std::runtime_error("Unsupported texture format!");
        }
    }

    void Texture::create(const TextureCreateInfo &info) {
        const VkFormat format = toVkFormat(info.texFormat);
        const VkExtent3D extent = {info.dataByLevel[0].width, info.dataByLevel[0].height, 1};
        const ImageCreateInfo imageCreateInfo {
            .device = info.device,
            .allocator = info.allocator,
            .flags = info.flags,
            .format = format,
            .extent = extent,
            .mipLevels = info.mipLevels,
            .arrayLayers = info.arrayLayers,
            .samples = info.samples,
            .tiling = info.tiling,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .memoryUsage = info.memoryUsage
        };
        mImage = Image(imageCreateInfo);

        const ImageViewCreateInfo imageViewCreateInfo {
            .device = info.device,
            .image = mImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevels = info.mipLevels,
            .baseMipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        };
        mView = ImageView(imageViewCreateInfo);

        const SamplerCreateInfo samplerCreateInfo {
            .device = info.device,
            .physicalDevice = info.physicalDevice,
            .addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .compareEnable = VK_FALSE,
            .mipLevels = info.mipLevels,
            .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
        };
        mSampler = Sampler(samplerCreateInfo);
    }

    void Texture::load(const TextureCreateInfo& info) {
        if (info.dataByLevel.empty()) {
            WARNING << "Failed to load texture, it is empty!";
            return;
        }
        // if (info.dataByLevel.size() > 1) {
        //     //mGenerateMipMap = false;
        //     mMipLevels = loadInfo.dataByLevel.size();
        // } else mGenerateMipMap = loadInfo.generateMipMap;

        //if (mGenerateMipMap) mMipLevels = calcMipLevels(mExtent.width, mExtent.height);
        for (size_t level = 0; level < info.dataByLevel.size(); ++level) {
            const auto& [data, width, height] = info.dataByLevel[level];
            load(info, data, {width, height}, level);
        }
    }

    static uint64_t formatToSize(const VkFormat format, const VkExtent2D extent) {
        uint64_t res = 0;
        switch (format) {
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
                res = ((extent.width + 3) / 4) * ((extent.height + 3) / 4) * 8;
                break;
            case VK_FORMAT_BC3_UNORM_BLOCK:
                res = ((extent.width + 3) / 4) * ((extent.height + 3) / 4) * 16;
                break;
            default:
                res = extent.width * extent.height * 4;
                break;
        }
        return res;
    }

    void Texture::load(const TextureCreateInfo& info, void* data,
                       const VkExtent2D extent, const uint32_t mipLevel) {
        auto [commandPool, commandBuffers] = beginCommandBuffer(info.device, info.queueFamilyIndex);
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const ImageTransitInfo transitInfo {
            .commandBuffer = commandBuffer,
            .image = mImage.get(),
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .baseMipLevel = mipLevel,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT
        };
        Image::transit(transitInfo);

        const VkFormat format = toVkFormat(info.texFormat);
        const uint32_t imageSize = formatToSize(format, extent);
        const BufferCreateInfo bufferCreateInfo {
            .allocator = info.allocator,
            .size = imageSize,
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        Buffer stagingBuffer = Buffer(bufferCreateInfo);
        const CopyDataToCPUBufferInfo copyDataToCpuBufferInfo {
            .data = data,
            .size = imageSize,
            .allocator = info.allocator,
            .allocation = stagingBuffer.allocation()
        };
        Buffer::copy(copyDataToCpuBufferInfo);
        const CopyBufferToImageInfo copyBufferToImageInfo {
            .commandBuffer = commandBuffer,
            .buffer = stagingBuffer.get(),
            .image = mImage.get(),
            .extent = extent,
            .mipLevel = mipLevel,
            .baseArrayLayer = 0,
            .layerCount = 1
        };
        Image::copy(copyBufferToImageInfo);
        endCommandBuffer(commandPool, commandBuffers, info.queue);
        //if (mGenerateMipMap) generateMipMaps();
    }

}
