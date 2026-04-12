//
// Created by igor on 4/7/26.
//

#include "Window.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    Window::Window(const WindowCreateInfo& createInfo):
    BaseWindow(createInfo.name.c_str(), createInfo.width, createInfo.height) {
        Window::init();
    }

    void Window::init() {
        if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
        glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

        mWindow = glfwCreateWindow(mWidth, mHeight, mName.c_str(), nullptr, nullptr);
        if (!mWindow) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
        glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    }
}
