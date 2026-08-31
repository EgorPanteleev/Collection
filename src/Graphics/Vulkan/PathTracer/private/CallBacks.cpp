//
// Created by igor on 6/8/26.
//

#include "CallBacks.hpp"

#include <imgui_internal.h>

namespace crv::graphics::vulkan {
    static bool inputBlocked() {
        return ImGui::GetTopMostPopupModal() != nullptr;
    }

    static PathTracerApp* appOf(GLFWwindow* window) {
        return static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
    }

    // Runs once per frame, after events are polled. Translates the accumulated
    // InputState into commands; the app drains and applies them by target.
    static void processInput(GLFWwindow* window, double) {
        if (inputBlocked()) return;
        auto app = appOf(window);
        const InputState& input = app->input();
        CommandStream& commands = app->commands();

        if (input.wasPressed(Key::Q)) commands.push(CommandType::QUIT);
        if (input.wasPressed(Key::Z)) commands.push(CommandType::TOGGLE_CONTROL_PANEL);
        if (input.wasPressed(Key::ESCAPE)) commands.push(CommandType::CLEAR_SELECTION);
        if (input.wasPressed(Key::X)) commands.push(CommandType::PICK_OBJECT);
        if (input.wasReleased(MouseButton::MIDDLE)) commands.push(CommandType::PICK_OBJECT);

        if (input.isPressed(Key::W)) commands.push(CommandType::MOVE_FORWARD);
        if (input.isPressed(Key::S)) commands.push(CommandType::MOVE_BACKWARD);
        if (input.isPressed(Key::A)) commands.push(CommandType::MOVE_LEFT);
        if (input.isPressed(Key::D)) commands.push(CommandType::MOVE_RIGHT);
        if (input.isPressed(Key::SPACE)) commands.push(CommandType::MOVE_UP);
        if (input.isPressed(Key::LEFT_CONTROL)) commands.push(CommandType::MOVE_DOWN);

        if (input.isPressed(Key::LEFT))  commands.push(CommandType::ROTATE_LEFT);
        if (input.isPressed(Key::RIGHT)) commands.push(CommandType::ROTATE_RIGHT);
        if (input.isPressed(Key::UP))    commands.push(CommandType::ROTATE_UP);
        if (input.isPressed(Key::DOWN))  commands.push(CommandType::ROTATE_DOWN);

        if (input.isPressed(MouseButton::RIGHT)) {
            const glm::dvec2 delta = input.cursorDelta();
            if (delta.x != 0.0 || delta.y != 0.0) commands.push(CommandType::LOOK);
        }
        if (input.scrollDelta().y != 0.0) commands.push(CommandType::ZOOM);
    }

    static void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_REPEAT) return;
        if (const auto mapped = Window::mapKey(key)) {
            appOf(window)->input().onKey(*mapped, action == GLFW_PRESS);
        }
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (const auto mapped = Window::mapMouseButton(button)) {
            appOf(window)->input().onMouseButton(*mapped, action == GLFW_PRESS);
        }
    }

    static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
        appOf(window)->input().onCursorPos(xpos, ypos);
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        appOf(window)->input().onScroll(xoffset, yoffset);
    }

    void setCallBacks(PathTracerApp* app) {
        Window& window = app->window();
        window.setUserPoint(app);
        window.makeContextCurrent();
        window.setKeyboardCallBack(processInput);
        window.setKeyCallBack(keyCallBack);
        window.setMouseButtonCallBack(mouseButtonCallback);
        window.setMouseMoveCallBack(mouseMoveCallback);
        window.setScrollCallBack(scrollCallback);
    }
}
