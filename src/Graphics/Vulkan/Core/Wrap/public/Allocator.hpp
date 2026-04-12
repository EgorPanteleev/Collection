//
// Created by igor on 4/12/26.
//

#ifndef COLLECTION_ALLOCATOR_HPP
#define COLLECTION_ALLOCATOR_HPP


#include <vk_mem_alloc.h>
#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct AllocatorCreateInfo {
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkInstance instance;
    };

    class Allocator: public DefaultWrapper<VmaAllocator> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Allocator(const AllocatorCreateInfo& info);
        Allocator& operator=(Allocator&&) = default;
        ~Allocator() override { Allocator::destroy(); }
        void destroy() override;
    protected:
    };
}

#endif //COLLECTION_ALLOCATOR_HPP