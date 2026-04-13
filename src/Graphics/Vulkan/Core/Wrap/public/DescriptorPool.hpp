//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_DESCRIPTORPOOL_HPP
#define COLLECTION_DESCRIPTORPOOL_HPP

#include <vulkan/vulkan_core.h>
#include "DefaultWrapper.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    struct DescriptorPoolCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkDescriptorPoolSize> poolSizes;
        uint32_t maxSets;
    };

    class DescriptorPool: public DefaultWrapper<VkDescriptorPool> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit DescriptorPool(const DescriptorPoolCreateInfo& info);
        DescriptorPool& operator=(DescriptorPool&&) = default;
        ~DescriptorPool() override { DescriptorPool::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_DESCRIPTORPOOL_HPP