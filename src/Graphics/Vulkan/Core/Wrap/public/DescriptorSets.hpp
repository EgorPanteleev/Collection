//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_DESCRIPTORSETS_HPP
#define COLLECTION_DESCRIPTORSETS_HPP

#include <vulkan/vulkan_core.h>
#include "DefaultWrapper.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    struct DescriptorSetsCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> layouts{};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::vector<uint32_t> variableCounts{};
    };

    struct DescriptorSetsUpdateInfo {
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
    };

    class DescriptorSets: public VectorWrapper<VkDescriptorSet> {
    public:
        using VectorWrapper::VectorWrapper;
        explicit DescriptorSets(const DescriptorSetsCreateInfo& info);
        DescriptorSets& operator=(DescriptorSets&&) = default;
        ~DescriptorSets() override { DescriptorSets::destroy(); }
        void destroy() override;
        void update(const DescriptorSetsUpdateInfo& info) const;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_DESCRIPTORSETS_HPP