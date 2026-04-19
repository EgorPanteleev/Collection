//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_FENCE_HPP
#define COLLECTION_FENCE_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct FenceCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
    };

    class Fence: public DefaultWrapper<VkFence> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Fence(const FenceCreateInfo& info);
        explicit Fence(Fence&&) = default;
        Fence& operator=(Fence&&) = default;
        ~Fence() override { Fence::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_FENCE_HPP