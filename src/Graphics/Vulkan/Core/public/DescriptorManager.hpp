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

    enum class BindingType {
        UBO,
        SSBO,
        AS,
        STORAGE_IMAGE,
        SAMPLED_IMAGE,
        TEXTURE
    };

    BindingDescription binding(BindingType type, VkShaderStageFlags stages, uint32_t count = 1);

    struct DescriptorBuildInfo {
        Context* context       = nullptr;
        uint32_t count         = 0;
        uint32_t variableCount = 1;
    };

    struct BufferResource {
        explicit BufferResource(const Buffer& buffer);
        VkBuffer     buffer = VK_NULL_HANDLE;
        VkDeviceSize size   = 0;
    };

    struct BufferArrayResource {
        explicit BufferArrayResource(const std::vector<Buffer>& buffers);
        std::vector<VkBuffer>     buffers{};
        std::vector<VkDeviceSize> sizes{};
    };

    struct ImageResource {
        explicit ImageResource(ImageView* view, VkImageLayout layout);
        explicit ImageResource(const Sampler& sampler, const ImageView& view, VkImageLayout layout);
        VkSampler     sampler   = VK_NULL_HANDLE;
        VkImageView   imageView = VK_NULL_HANDLE;
        VkImageLayout layout    = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct ImageArrayResource {
        explicit ImageArrayResource(const std::vector<Sampler>& samplers, const std::vector<ImageView>& views,
            const std::vector<VkImageLayout>& layouts);
        explicit ImageArrayResource(std::vector<TexturesByType>& textures);
        explicit ImageArrayResource(std::vector<Texture>& textures);
        std::vector<VkSampler>     samplers{};
        std::vector<VkImageView>   imageViews{};
        std::vector<VkImageLayout> layouts{};
    };

    struct ASResource {
        explicit ASResource(VkAccelerationStructureKHR as): accelerationStructures({as}) {}
        explicit ASResource(const std::vector<VkAccelerationStructureKHR>& structures): accelerationStructures(structures) {}
        std::vector<VkAccelerationStructureKHR> accelerationStructures;
    };

    class DescriptorManager {
    public:
        using Resource = std::variant<BufferResource, BufferArrayResource, ImageResource, ImageArrayResource, ASResource>;
        void add(BindingType type, VkShaderStageFlags stages, uint32_t count = 1) { mBindings.push_back(binding(type, stages, count)); }
        void add(const BindingDescription& desc) { mBindings.push_back(desc); }
        void bind(uint32_t setIndex, const Resource& resource) { mResources[setIndex].emplace_back(resource); }
        void bind(uint32_t binding, uint32_t setIndex, const Resource& resource) { mResources[setIndex][binding] = resource; }
        void bind(uint32_t binding, uint32_t setIndex, uint32_t arrayIndex, const ImageResource& image);
        void bind(uint32_t binding, uint32_t setIndex, uint32_t arrayIndex, const BufferResource& buffer);
        void build(const DescriptorBuildInfo& info);
        void update();
        void update(uint32_t binding, uint32_t setIndex);
        void update(uint32_t binding, uint32_t setIndex, uint32_t arrayIndex);
        VkDescriptorSet& set(const uint32_t index) { return mDescriptorSets[index]; }
        [[nodiscard]] std::vector<VkDescriptorSetLayout> layouts(uint32_t count) const;
    private:
        void createLayout(const DescriptorBuildInfo& info);
        void createPool(const DescriptorBuildInfo& info);
        void createSets(const DescriptorBuildInfo& info);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, uint32_t setIndex);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const BufferResource& resource);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const BufferArrayResource& resource);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const ImageResource& resource );
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const ImageArrayResource& resource);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, const ASResource& resource );
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, uint32_t setIndex, uint32_t arrayIndex);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, uint32_t arrayIndex, const ImageArrayResource& resource);
        VkWriteDescriptorSet getDescriptorWrite(uint32_t binding, uint32_t arrayIndex, const BufferArrayResource& resource);

        using Resources = std::unordered_map<uint32_t, std::vector<Resource>>;
        std::vector<BindingDescription>    mBindings{};
        Resources                          mResources{};
        DescriptorSetLayout                mDescriptorSetLayout{};
        DescriptorPool                     mDescriptorPool{};
        DescriptorSets                     mDescriptorSets{};

        std::vector<VkDescriptorBufferInfo>                        mBufferInfos;
        std::vector<VkDescriptorImageInfo>                         mImageInfos;
        std::vector<VkWriteDescriptorSetAccelerationStructureKHR>  mASInfos;
    };
}

#endif //COLLECTION_DESCRIPTORSCHEMA_HPP