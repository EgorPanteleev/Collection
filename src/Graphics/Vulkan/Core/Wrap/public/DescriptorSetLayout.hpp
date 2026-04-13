//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_DESCRIPTORSET_HPP
#define COLLECTION_DESCRIPTORSET_HPP

#include <vulkan/vulkan_core.h>
#include "DefaultWrapper.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    struct DescriptorSetLayoutCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayoutBinding> bindings{};
        std::vector<VkDescriptorBindingFlags> bindingFlags{};
    };

    class DescriptorSetLayout: public DefaultWrapper<VkDescriptorSetLayout> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit DescriptorSetLayout(const DescriptorSetLayoutCreateInfo& info);
        DescriptorSetLayout& operator=(DescriptorSetLayout&&) = default;
        ~DescriptorSetLayout() override { DescriptorSetLayout::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_DESCRIPTORSET_HPP