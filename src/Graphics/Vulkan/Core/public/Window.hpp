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

namespace crv::graphics::vulkan {
    struct WindowCreateInfo {
        int width = 800;
        int height = 600;
        std::string name = "Vulkan Window";
    };

    class Window: public BaseWindow {
    public:
        using BaseWindow::BaseWindow;
        explicit Window(const WindowCreateInfo& createInfo);
        Window& operator=(Window&&) noexcept = default;
        ~Window() override = default;
    protected:
        void init() override;
    };
}

#endif //COLLECTION_WINDOW_HPP