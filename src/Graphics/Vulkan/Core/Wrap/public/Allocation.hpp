//
// Created by igor on 4/15/26.
//

#ifndef COLLECTION_ALLOCATION_HPP
#define COLLECTION_ALLOCATION_HPP

#include <vk_mem_alloc.h>
#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    class Allocation: public DefaultWrapper<VmaAllocation> {
    public:
        using DefaultWrapper::DefaultWrapper;
        Allocation() = default;
        Allocation& operator=(Allocation&&) = default;
        ~Allocation() override { Allocation::destroy(); }
        void destroy() override {}
    protected:
    };
}

#endif //COLLECTION_ALLOCATION_HPP