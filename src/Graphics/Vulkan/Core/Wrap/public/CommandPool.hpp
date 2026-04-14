//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_COMMANDPOOL_HPP
#define COLLECTION_COMMANDPOOL_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct CommandPoolCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        VkCommandPoolCreateFlags flags;
        uint32_t queueFamilyIndex;
    };

    class CommandPool: public DefaultWrapper<VkCommandPool> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit CommandPool(const CommandPoolCreateInfo& info);
        CommandPool& operator=(CommandPool&&) = default;
        ~CommandPool() override { CommandPool::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_COMMANDPOOL_HPP