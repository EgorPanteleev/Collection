//
// Created by igor on 4/11/26.
//

#ifndef COLLECTION_SURFACE_HPP
#define COLLECTION_SURFACE_HPP

#include <vulkan/vulkan_core.h>
#include "DefaultWrapper.hpp"

class GLFWwindow;
namespace crv::graphics::vulkan {
    struct SurfaceCreateInfo {
        VkInstance instance;
        GLFWwindow* window;
    };

    class Surface: public DefaultWrapper<VkSurfaceKHR> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Surface(const SurfaceCreateInfo& info);
        Surface& operator=(Surface&&) = default;
        ~Surface() override { Surface::destroy(); }
        void destroy() override;
    protected:
        VkInstance mInstance = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_SURFACE_HPP