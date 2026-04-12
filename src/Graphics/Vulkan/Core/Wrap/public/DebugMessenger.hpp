//
// Created by igor on 4/12/26.
//

#ifndef COLLECTION_DEBUGMESSENGER_HPP
#define COLLECTION_DEBUGMESSENGER_HPP

#include <vulkan/vulkan_core.h>
#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct DebugMessengerCreateInfo {
        VkInstance instance;
    };

    class DebugMessenger: public DefaultWrapper<VkDebugUtilsMessengerEXT> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit DebugMessenger(const DebugMessengerCreateInfo& info);
        DebugMessenger& operator=(DebugMessenger&&) = default;
        ~DebugMessenger() override { DebugMessenger::destroy(); }
        void destroy() override;
    protected:
        VkInstance mInstance = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_DEBUGMESSENGER_HPP