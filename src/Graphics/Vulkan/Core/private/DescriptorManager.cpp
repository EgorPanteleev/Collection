//
// Created by igor on 5/23/26.
//

#include "DescriptorManager.hpp"

namespace crv::graphics::vulkan {
    static VkDescriptorType toVkDescriptorType(BindingType type) {
        switch (type) {
            case BindingType::UBO:
                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            case BindingType::SSBO:
                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case BindingType::AS:
                return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            case BindingType::STORAGE_IMAGE:
                return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            case BindingType::SAMPLED_IMAGE:
            case BindingType::TEXTURE:
                return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
    }

    BindingDescription binding(BindingType type, VkShaderStageFlags stages, uint32_t count) {
        switch (type) {
            case BindingType::TEXTURE:
                return {
                    .type = toVkDescriptorType(type),
                    .stages = stages,
                    .flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
                    .count = count
                };
            default:
                return {
                    .type = toVkDescriptorType(type),
                    .stages = stages,
                    .count = count
                };
        }
    }

    BufferResource::BufferResource(const Buffer& buffer): buffer(buffer.get()), size(buffer.size()) {}

    BufferArrayResource::BufferArrayResource(const std::vector<Buffer>& buffers_) {
        for (const auto& buffer: buffers_) {
            buffers.push_back(buffer.get());
            sizes.push_back(buffer.size());
        }
    }

    ImageResource::ImageResource(ImageView* view, const VkImageLayout layout):
        sampler(VK_NULL_HANDLE), imageView(view->get()), layout(layout) {}

    ImageResource::ImageResource(const Sampler& sampler, const ImageView& view, VkImageLayout layout):
        sampler(sampler.get()), imageView(view.get()), layout(layout) {}

    ImageArrayResource::ImageArrayResource(const std::vector<Sampler>& samplers_, const std::vector<ImageView>& views,
        const std::vector<VkImageLayout>& layouts_): layouts(layouts_) {
        for (const auto& s: samplers_) samplers.push_back(s.get());
        for (const auto& view: views) imageViews.push_back(view.get());
    }

    ImageArrayResource::ImageArrayResource(std::vector<TexturesByType>& textures) {
        for (size_t i = 0; i < textures.size(); i++) {
            auto& texturesByType = textures[i];
            for (int j = 0; j < cm::Texture::Type::UNKNOWN; ++j) {
                auto& texture = texturesByType[j];
                if (texture.view() == VK_NULL_HANDLE) continue;
                samplers.push_back(texture.sampler());
                imageViews.push_back(texture.view());
                layouts.push_back(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    ImageArrayResource::ImageArrayResource(std::vector<Texture>& textures) {
        for (size_t i = 0; i < textures.size(); i++) {
            auto& texture = textures[i];
            if (texture.view() == VK_NULL_HANDLE) continue;
            samplers.push_back(texture.sampler());
            imageViews.push_back(texture.view());
            layouts.push_back(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    void DescriptorManager::bind(const uint32_t binding, const uint32_t setIndex, const uint32_t arrayIndex,
        const ImageResource& image) {
        auto& array = std::get<ImageArrayResource>(mResources[setIndex][binding]);
        if (arrayIndex >= array.imageViews.size()) {
            array.samplers.resize(arrayIndex + 1, VK_NULL_HANDLE);
            array.imageViews.resize(arrayIndex + 1, VK_NULL_HANDLE);
            array.layouts.resize(arrayIndex + 1, VK_IMAGE_LAYOUT_UNDEFINED);
        }
        array.samplers[arrayIndex]   = image.sampler;
        array.imageViews[arrayIndex] = image.imageView;
        array.layouts[arrayIndex]    = image.layout;
    }

    void DescriptorManager::bind(const uint32_t binding, const uint32_t setIndex, const uint32_t arrayIndex,
        const BufferResource& buffer) {
        auto& array = std::get<BufferArrayResource>(mResources[setIndex][binding]);
        if (arrayIndex >= array.buffers.size()) {
            array.buffers.resize(arrayIndex + 1, VK_NULL_HANDLE);
            array.sizes.resize(arrayIndex + 1, 0);
        }
        array.buffers[arrayIndex] = buffer.buffer;
        array.sizes[arrayIndex]   = buffer.size;
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
        mASInfos.reserve(resourcesCount * mDescriptorSets.size());
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
        for (uint32_t setIndex = 0; setIndex < mDescriptorSets.size(); ++setIndex) {
            std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.reserve(mBindings.size());
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

    void DescriptorManager::update(const uint32_t binding, const uint32_t setIndex) {
        mBufferInfos.clear();
        mImageInfos.clear();
        mASInfos.clear();
        mBufferInfos.reserve(mBindings[binding].count);
        mImageInfos.reserve(mBindings[binding].count);
        mASInfos.reserve(mBindings[binding].count);
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites(mDescriptorSets.size());
        descriptorsWrites[setIndex].push_back(getDescriptorWrite(binding, setIndex));
        mDescriptorSets.update({.descriptorsWrites = descriptorsWrites});
    }

    void DescriptorManager::update(const uint32_t binding, const uint32_t setIndex, const uint32_t arrayIndex) {
        mBufferInfos.clear();
        mImageInfos.clear();
        const VkWriteDescriptorSet write = getDescriptorWrite(binding, setIndex, arrayIndex);
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites(mDescriptorSets.size());
        descriptorsWrites[setIndex].push_back(write);
        mDescriptorSets.update({.descriptorsWrites = descriptorsWrites});
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
        return std::visit([this, binding](auto&& value) {
            return getDescriptorWrite(binding, value);
        }, resource); 
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const BufferResource& resource) {
        const uint32_t baseBufferInfo = mBufferInfos.size();
        mBufferInfos.push_back({.buffer = resource.buffer, .offset = 0, .range = resource.size});
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = mBindings[binding].type,
            .pImageInfo = nullptr,
            .pBufferInfo = &mBufferInfos[baseBufferInfo],
            .pTexelBufferView = nullptr
        };
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const BufferArrayResource& resource) {
        const uint32_t baseBufferInfo = mBufferInfos.size();
        for (size_t i = 0; i < resource.buffers.size(); ++i) {
            mBufferInfos.push_back({.buffer = resource.buffers[i], .offset = 0, .range = resource.sizes[i]});
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
        mImageInfos.push_back({.sampler = resource.sampler, .imageView = resource.imageView, .imageLayout = resource.layout});
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = mBindings[binding].type,
            .pImageInfo = &mImageInfos[baseImageInfo]
        };
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const ImageArrayResource& resource) {
        const uint32_t baseImageInfo = mImageInfos.size();
        for (size_t i = 0; i < resource.imageViews.size(); ++i) {
            mImageInfos.push_back({.sampler = resource.samplers[i], .imageView = resource.imageViews[i], .imageLayout = resource.layouts[i]});
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

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const uint32_t arrayIndex, const ImageArrayResource& resource) {
        const uint32_t baseImageInfo = mImageInfos.size();
        mImageInfos.push_back({.sampler = resource.samplers[arrayIndex], .imageView = resource.imageViews[arrayIndex], .imageLayout = resource.layouts[arrayIndex]});
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = arrayIndex,
            .descriptorCount = 1,
            .descriptorType = mBindings[binding].type,
            .pImageInfo = &mImageInfos[baseImageInfo]
        };
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const uint32_t setIndex, const uint32_t arrayIndex) {
        const Resource& resource = mResources[setIndex][binding];
        return std::visit([this, binding, arrayIndex](auto&& value) {
            using Type = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Type, BufferArrayResource> or
                          std::is_same_v<Type, ImageArrayResource>) return getDescriptorWrite(binding, arrayIndex, value);
            return VkWriteDescriptorSet{};
        }, resource);
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(const uint32_t binding, const uint32_t arrayIndex, const BufferArrayResource& resource) {
        const uint32_t baseBufferInfo = mBufferInfos.size();
        mBufferInfos.push_back({.buffer = resource.buffers[arrayIndex], .offset = 0, .range = resource.sizes[arrayIndex]});
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = arrayIndex,
            .descriptorCount = 1,
            .descriptorType = mBindings[binding].type,
            .pImageInfo = nullptr,
            .pBufferInfo = &mBufferInfos[baseBufferInfo],
            .pTexelBufferView = nullptr
        };
    }

    VkWriteDescriptorSet DescriptorManager::getDescriptorWrite(uint32_t binding, const ASResource& resource ) {
        const uint32_t baseASInfo = mASInfos.size();
        for (int i = 0; i < resource.accelerationStructures.size(); ++i) {
            VkWriteDescriptorSetAccelerationStructureKHR info{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &resource.accelerationStructures[i]
            };
            mASInfos.push_back(info);
        }
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &mASInfos[baseASInfo],
            .dstBinding = binding,
            .descriptorCount = static_cast<uint32_t>(resource.accelerationStructures.size()),
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
        };
    }
}
