//
// Created by igor on 4/11/26.
//

#ifndef COLLECTION_INSTANCE_HPP
#define COLLECTION_INSTANCE_HPP

#include "DefaultWrapper.hpp"
#include "Message.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    struct InstanceCreateInfo {
        std::vector<const char *> validationLayers{};
        bool enableValidationLayers = false;
    };

    class Instance: public DefaultWrapper<VkInstance> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Instance(const InstanceCreateInfo& info);
        Instance& operator=(Instance&&) = default;
        ~Instance() override { Instance::destroy(); }
        void destroy() override;
        static bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers);
        static std::vector<const char *> getRequiredExtensions(bool enableValidationLayers);
        static const VkDebugUtilsMessengerCreateInfoEXT createDebugMessengerCreateInfo();
        static void checkGlfwRequiredInstanceExtensions(bool enableValidationLayers);
    protected:

    };
}

#endif //COLLECTION_INSTANCE_HPP