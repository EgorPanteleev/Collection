//
// Created by igor on 10/18/25.
//

#include "Window.hpp"

namespace crv::graphics {
    Window::Window(const char* name, int width, int height): mWindow(nullptr) {
        initWindow(name, width, height);
    }

    Window::~Window() {
        if (mWindow) {
            glfwDestroyWindow(mWindow);
        }
        glfwTerminate();
    }

    void Window::initWindow(const char* name, int width, int height) {
        if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        mWindow = glfwCreateWindow(width, height, name, nullptr, nullptr);
        if (!mWindow) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
    }
}