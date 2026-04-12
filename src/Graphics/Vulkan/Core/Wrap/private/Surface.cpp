//
// Created by igor on 4/11/26.
//

#include "Surface.hpp"
#include "Message.hpp"

#include <stdexcept>
#include <GL/glew.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace crv::graphics::vulkan {
    Surface::Surface(const SurfaceCreateInfo& info): mInstance(info.instance) {
        if (glfwCreateWindowSurface(mInstance, info.window, nullptr, &mHandle) != VK_SUCCESS) {
            const char* desc;
            int code = glfwGetError(&desc);
            ERROR << desc;
            throw std::runtime_error("Failed to create window surface!");
        }
        INFO << "Window surface created!";
    }

    void Surface::destroy() {
        if (mHandle == VK_NULL_HANDLE) return;
        vkDestroySurfaceKHR(mInstance, mHandle, nullptr);
    }
}
