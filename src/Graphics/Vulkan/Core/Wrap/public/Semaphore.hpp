//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_SEMAPHORE_HPP
#define COLLECTION_SEMAPHORE_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct SemaphoreCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
    };

    class Semaphore: public DefaultWrapper<VkSemaphore> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Semaphore(const SemaphoreCreateInfo& info);
        Semaphore& operator=(Semaphore&&) = default;
        ~Semaphore() override { Semaphore::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_SEMAPHORE_HPP