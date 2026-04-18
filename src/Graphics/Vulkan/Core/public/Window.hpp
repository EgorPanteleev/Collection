//
// Created by igor on 4/7/26.
//

#ifndef COLLECTION_WINDOW_HPP
#define COLLECTION_WINDOW_HPP

#include <GL/glew.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "BaseWindow.hpp"
#include "DefaultWrapper.hpp"
#include <functional>

namespace crv::graphics::vulkan {
    struct WindowCreateInfo {
        int width = 800;
        int height = 600;
        std::string name = "Vulkan Window";
    };

    class Window: public BaseWindow {
    public:
        using BaseWindow::BaseWindow;
        using KeyboardCallBack = std::function<void(GLFWwindow*, void*, double)>;
        explicit Window(const WindowCreateInfo& createInfo);
        Window& operator=(Window&&) noexcept = default;
        ~Window() override = default;
        void setKeyboardCallBack(const KeyboardCallBack& callBack) { mKeyboardCallBack = callBack; }
        void keyboardCallBack(void* camera, double delta) const { mKeyboardCallBack(glfwWindow(), camera, delta); }
    protected:
        void init() override;
        KeyboardCallBack mKeyboardCallBack;
    };
}

#endif //COLLECTION_WINDOW_HPP