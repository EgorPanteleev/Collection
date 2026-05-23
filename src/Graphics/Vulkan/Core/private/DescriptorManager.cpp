//
// Created by igor on 5/23/26.
//

#include "DescriptorManager.hpp"

namespace crv::graphics::vulkan {
    BufferResource::BufferResource(VkBuffer buffer, VkDeviceSize size): buffers({buffer}), sizes({size}) {}
    BufferResource::BufferResource(const Buffer& buffer): BufferResource(buffer.get(), buffer.size()) {}
    BufferResource::BufferResource(const std::vector<Buffer>& buffers_) {
        for (const auto& buffer: buffers_) {
            buffers.push_back(buffer.get());
            sizes.push_back(buffer.size());
        }
    }

    ImageResource::ImageResource(VkSampler sampler, VkImageView view, VkImageLayout layout):
        samplers({sampler}), imageViews({view}), layouts({layout}) {}

    ImageResource::ImageResource(VkSampler sampler, const ImageView& view, VkImageLayout layout):
        ImageResource(sampler, view.get(), layout) {}

    ImageResource::ImageResource(const std::vector<VkSampler>& samplers, const std::vector<ImageView>& views,
        const std::vector<VkImageLayout>& layouts): samplers(samplers), layouts(layouts){
        for (const auto& view: views) imageViews.push_back(view.get());
    }

    ImageResource::ImageResource(std::vector<TexturesByType>& mTextures) {
        for (size_t i = 0; i < mTextures.size(); i++) {
            auto& texturesByType = mTextures[i];
            for (int j = 0; j < cm::Texture::Type::UNKNOWN; ++j) {
                auto& texture = texturesByType[j];
                if (texture.view() == VK_NULL_HANDLE) continue;
                samplers.push_back(texture.sampler());
                imageViews.push_back(texture.view());
                layouts.push_back(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    void DescriptorManager::build(const DescriptorBuildInfo& info) {
        createLayout(info);
        createPool(info);
        createSets(info);
    }

    void DescriptorManager::update() {
        mBufferInfos.clear();
        mImageInfos.clear();
        uint32_t resourcesCount = 0;
        for (uint32_t bindingIndex = 0; bindingIndex < mBindings.size(); ++bindingIndex)
            resourcesCount += mBindings[bindingIndex].count;

        mBufferInfos.reserve(resourcesCount * mDescriptorSets.size());
        mImageInfos.reserve(resourcesCount * mDescriptorSets.size());
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
        for (uint32_t setIndex = 0; setIndex < mDescriptorSets.size(); ++setIndex) {
            std::vector<VkWriteDescriptorSet> descriptorWrites;
            for (uint32_t bindingIndex = 0; bindingIndex < mBindings.size(); ++bindingIndex) {
                descriptorWrites.push_back(getDescriptorWrite(bindingIndex, setIndex));
            }
            descriptorsWrites.push_back(descriptorWrites);
        }
        const DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
    }

    void DescriptorManager::update(const uint32_t binding, const uint32_t setIndex, const BufferResource& resource) {
        mResources[setIndex][binding] = resource;
    }

    void DescriptorManager::update(const uint32_t binding, const uint32_t setIndex, const ImageResource& resource) {
        mResources[setIndex][binding] = resource;
    }

    void DescriptorManager::update(uint32_t binding, const BufferResource& resource) {
        for (uint32_t setIndex = 0; setIndex < mDescriptorSets.size(); ++setIndex) {
            mResources[setIndex][binding] = resource;
        }
    }

    void DescriptorManager::update(uint32_t binding, const ImageResource& resource) {
        for (uint32_t setIndex = 0; setIndex < mDescriptorSets.size(); ++setIndex) {
            mResources[setIndex][binding] = resource;
        }
    }

    void DescriptorManager::createLayout(const DescriptorBuildInfo& info) {
        std::vector <VkDescriptorSetLayoutBinding> bindings;
        std::vector<VkDescriptorBindingFlags> bindingsFlags;
        for (uint32_t bindingIndex = 0; bindingIndex < mBindings.size(); ++bindingIndex) {
            const BindingDescription& binding = mBindings[bindingIndex];
            VkDescriptorSetLayoutBinding layoutBinding {
                .binding = bindingIndex,
                .descriptorType = binding.type,
                .descriptorCount = binding.count,
                .stageFlags = binding.stages,
            };
            bindings.push_back(layoutBinding);
            bindingsFlags.push_back(binding.flags);
        }
        const DescriptorSetLayoutCreateInfo createInfo {
            .device = info.context->device(),
            .bindings = bindings,
            .bindingFlags = bindingsFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }

    void DescriptorManager::createPool(const DescriptorBuildInfo& info) {
        std::vector<VkDescriptorPoolSize> poolSizes;
        for (uint32_t bindingIndex = 0; bindingIndex < mBindings.size(); ++bindingIndex) {
            const BindingDescription& binding = mBindings[bindingIndex];
            poolSizes.emplace_back(binding.type, info.count * binding.count);
        }

        const DescriptorPoolCreateInfo createInfo {
            .device = info.context->device(),
            .poolSizes = poolSizes,
            .maxSets = info.count
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    std::vector<VkDescriptorSetLayout> DescriptorManager::layouts(const uint32_t count) const {
        std::vector<VkDescriptorSetLayout> layouts;
        for (uint32_t i = 0; i < count; ++i) {
            layouts.push_back(mDescriptorSetLayout.get());
        }
        return layouts;
    }

    void DescriptorManager::createSets(const DescriptorBuildInfo& info) {
        const std::vector variableCounts(info.count, info.variableCount);
        const DescriptorSetsCreateInfo createInfo {
            .device = info.context->device(),
            .layouts = layouts(info.count),
            .pool = mDescriptorPool.get(),
            .variableCounts = variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const uint32_t setIndex) {
        const Resource& resource = mResources[setIndex][binding];
        if (std::holds_alternative<BufferResource>(resource)) {
            return getDescriptorWrite(binding, std::get<BufferResource>(resource));
        }
        return getDescriptorWrite(binding, std::get<ImageResource>(resource));
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const BufferResource& resource) {
        const uint32_t baseBufferInfo = mBufferInfos.size();
        for (int i = 0; i < resource.buffers.size(); ++i) {
            const VkDescriptorBufferInfo info {
                .buffer = resource.buffers[i],
                .offset = 0,
                .range = resource.sizes[i]
            };
            mBufferInfos.push_back(info);
        }
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = static_cast<uint32_t>(resource.buffers.size()),
            .descriptorType = mBindings[binding].type,
            .pImageInfo = nullptr,
            .pBufferInfo = &mBufferInfos[baseBufferInfo],
            .pTexelBufferView = nullptr
        };
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const ImageResource& resource) {
        const uint32_t baseImageInfo = mImageInfos.size();
        for (int i = 0; i < resource.imageViews.size(); ++i) {
            VkDescriptorImageInfo info {
                .sampler = resource.samplers[i],
                .imageView = resource.imageViews[i],
                .imageLayout = resource.layouts[i]
            };
            mImageInfos.push_back(info);
        }
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = static_cast<uint32_t>(resource.imageViews.size()),
            .descriptorType = mBindings[binding].type,
            .pImageInfo = &mImageInfos[baseImageInfo]
        };
    }
}
