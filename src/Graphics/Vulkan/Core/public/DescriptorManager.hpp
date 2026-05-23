//
// Created by igor on 5/23/26.
//

#ifndef COLLECTION_DESCRIPTORSCHEMA_HPP
#define COLLECTION_DESCRIPTORSCHEMA_HPP

#include <vector>
#include <deque>
#include <variant>
#include <unordered_map>

#include "CoreUtils.hpp"
#include "DescriptorSetLayout.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSets.hpp"
#include "Texture.hpp"

namespace crv::graphics::vulkan {
    struct BindingDescription {
        VkDescriptorType         type   = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        VkShaderStageFlags       stages = VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
        VkDescriptorBindingFlags flags  = 0;
        uint32_t                 count  = 1;
    };

    struct DescriptorBuildInfo {
        Context* context       = nullptr;
        uint32_t count         = 0;
        uint32_t variableCount = 1;
    };

    struct BufferResource {
        explicit BufferResource(VkBuffer buffer, VkDeviceSize size);
        explicit BufferResource(const Buffer& buffer);
        explicit BufferResource(const std::vector<Buffer>& buffers_);
        std::vector<VkBuffer>     buffers{};
        std::vector<VkDeviceSize> sizes{};
    };

    struct ImageResource {
        explicit ImageResource(VkSampler sampler, VkImageView view, VkImageLayout layout);
        explicit ImageResource(VkSampler sampler, const ImageView& view, VkImageLayout layout);
        explicit ImageResource(const std::vector<VkSampler>& samplers, const std::vector<ImageView>& views,
            const std::vector<VkImageLayout>& layouts);
        explicit ImageResource(std::vector<TexturesByType>& mTextures);
        std::vector<VkSampler>     samplers{};
        std::vector<VkImageView>   imageViews{};
        std::vector<VkImageLayout> layouts{};
    };

    class DescriptorManager {
    public:
        using Resource = std::variant<BufferResource, ImageResource>;
        void add(const BindingDescription& desc) { mBindings.push_back(desc); }
        void add(uint32_t setIndex, const BufferResource& resource) { mResources[setIndex].emplace_back(resource); }
        void add(uint32_t setIndex, const ImageResource& resource)  { mResources[setIndex].emplace_back(resource); }
        void update(uint32_t binding, uint32_t setIndex, const BufferResource& resource);
        void update(uint32_t binding, uint32_t setIndex, const ImageResource& resource);
        void update(uint32_t binding, const BufferResource& resource);
        void update(uint32_t binding, const ImageResource& resource);
        void build(const DescriptorBuildInfo& info);
        void update();
        VkDescriptorSet& set(const uint32_t index) { return mDescriptorSets[index]; }
        [[nodiscard]] std::vector<VkDescriptorSetLayout> layouts(uint32_t count) const;
    private:
        void createLayout(const DescriptorBuildInfo& info);
        void createPool(const DescriptorBuildInfo& info);
        void createSets(const DescriptorBuildInfo& info);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, uint32_t setIndex);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const BufferResource& resource);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const ImageResource& resource );

        using Resources = std::unordered_map<uint32_t, std::vector<Resource>>;
        std::vector<BindingDescription>    mBindings{};
        Resources                          mResources{};
        DescriptorSetLayout                mDescriptorSetLayout{};
        DescriptorPool                     mDescriptorPool{};
        DescriptorSets                     mDescriptorSets{};

        std::vector<VkDescriptorBufferInfo> mBufferInfos;
        std::vector<VkDescriptorImageInfo>  mImageInfos;
    };
}

#endif //COLLECTION_DESCRIPTORSCHEMA_HPP