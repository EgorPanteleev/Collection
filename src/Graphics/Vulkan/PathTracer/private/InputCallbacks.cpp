//
// Created by igor on 6/8/26.
//

#include "InputCallbacks.hpp"

namespace crv::graphics::vulkan {
    static InputState* inputOf(GLFWwindow* window) {
        return static_cast<InputState*>(glfwGetWindowUserPointer(window));
    }

    static void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_REPEAT) return;
        if (const auto mapped = Window::mapKey(key)) {
            inputOf(window)->onKey(*mapped, action == GLFW_PRESS);
        }
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (const auto mapped = Window::mapMouseButton(button)) {
            inputOf(window)->onMouseButton(*mapped, action == GLFW_PRESS);
        }
    }

    static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
        inputOf(window)->onCursorPos(xpos, ypos);
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        inputOf(window)->onScroll(xoffset, yoffset);
    }

    void setCallBacks(Window& window, InputState& input) {
        window.setUserPoint(&input);
        window.makeContextCurrent();
        window.setKeyCallBack(keyCallBack);
        window.setMouseButtonCallBack(mouseButtonCallback);
        window.setMouseMoveCallBack(mouseMoveCallback);
        window.setScrollCallBack(scrollCallback);
    }
}
