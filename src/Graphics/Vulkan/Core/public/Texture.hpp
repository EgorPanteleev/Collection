//
// Created by igor on 4/19/26.
//

#ifndef COLLECTION_TEXTURE_HPP
#define COLLECTION_TEXTURE_HPP

#include "Image.hpp"
#include "ImageView.hpp"
#include "Sampler.hpp"
#include "Material.hpp"

namespace crv::graphics::vulkan {
    namespace cm = model;
    using cmt = model::Texture;
    struct TextureCreateInfo {
        VkDevice                    device           = VK_NULL_HANDLE;
        VkPhysicalDevice            physicalDevice   = VK_NULL_HANDLE;
        VmaAllocator                allocator        = VK_NULL_HANDLE;
        VkQueue                     queue            = VK_NULL_HANDLE;
        uint32_t                    queueFamilyIndex = 0;
        std::vector<cmt::LevelData> dataByLevel{};
        cmt::Format                 texFormat        = cm::Texture::Format::UNDEFINED;
        VkImageCreateFlags          flags            = 0;
        uint32_t                    mipLevels        = 1;
        uint32_t                    arrayLayers      = 1;
        VkSampleCountFlagBits       samples          = VK_SAMPLE_COUNT_1_BIT;
        VkImageTiling               tiling           = VK_IMAGE_TILING_OPTIMAL;
        VmaMemoryUsage              memoryUsage      = VMA_MEMORY_USAGE_AUTO;
    };

    class Texture {
    public:
        Texture() = default;
        explicit Texture(const TextureCreateInfo& info);
        Texture(Texture&&) = default;
        Texture& operator=(Texture&&) = default;
        VkImage image() { return mImage.get(); }
        VkImageView view() { return mView.get(); }
        VkSampler sampler() { return mSampler.get(); }
    protected:
        void create(const TextureCreateInfo& info);
        void load(const TextureCreateInfo& info);
        void load(const TextureCreateInfo& info, void* data,
        VkExtent2D extent, uint32_t mipLevel);
        Image mImage{};
        ImageView mView{};
        Sampler mSampler{};
    };

    struct TexturesByType {
        Texture& operator[](const int type) { return mTexturesByType[type]; }
        std::array<Texture, cm::Texture::UNKNOWN> mTexturesByType{};
    };
}

#endif //COLLECTION_TEXTURE_HPP