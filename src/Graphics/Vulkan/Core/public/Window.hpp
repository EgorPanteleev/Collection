//
// Created by igor on 4/7/26.
//

#ifndef COLLECTION_WINDOW_HPP
#define COLLECTION_WINDOW_HPP

#include <GL/glew.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "BaseWindow.hpp"
#include "InputState.hpp"
#include "DefaultWrapper.hpp"
#include <functional>
#include <optional>

namespace crv::graphics::vulkan {
    struct WindowCreateInfo {
        int width = 800;
        int height = 600;
        std::string name = "Vulkan Window";
    };

    class Window: public BaseWindow {
    public:
        using BaseWindow::BaseWindow;
        using KeyboardCallBack = std::function<void(GLFWwindow*, double)>;
        explicit Window(const WindowCreateInfo& createInfo);
        Window& operator=(Window&&) noexcept = default;
        ~Window() override = default;
        void setKeyboardCallBack(const KeyboardCallBack& callBack) { mKeyboardCallBack = callBack; }
        void keyboardCallBack(double delta) const { mKeyboardCallBack(glfwWindow(), delta); }

        [[nodiscard]] static std::optional<Key> mapKey(int glfwKey);
        [[nodiscard]] static std::optional<MouseButton> mapMouseButton(int glfwButton);
    protected:
        void init() override;
        KeyboardCallBack mKeyboardCallBack;
    };
}

#endif //COLLECTION_WINDOW_HPP