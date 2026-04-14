//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_COMMANDBUFFERS_HPP
#define COLLECTION_COMMANDBUFFERS_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct CommandBuffersCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        uint32_t bufferCount = 0;
    };

    class CommandBuffers: public VectorWrapper<VkCommandBuffer> {
    public:
        using VectorWrapper::VectorWrapper;
        explicit CommandBuffers(const CommandBuffersCreateInfo& info);
        CommandBuffers& operator=(CommandBuffers&&) = default;
        ~CommandBuffers() override { CommandBuffers::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
        VkCommandPool mCommandPool = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_COMMANDBUFFERS_HPP