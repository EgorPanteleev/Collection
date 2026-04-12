//
// Created by igor on 10/18/25.
//

#include "Window.hpp"

#include <stdexcept>

namespace crv::graphics {
    void Window::init() {
        if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        mWindow = glfwCreateWindow(mWidth, mHeight, mName.c_str(), nullptr, nullptr);
        if (!mWindow) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
    }
}