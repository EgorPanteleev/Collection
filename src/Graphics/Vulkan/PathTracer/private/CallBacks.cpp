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

    static void objectSelected(PathTracerApp* app, bool additive) {
        const glm::dvec2 cursor = app->input().cursorPos();
        GLFWwindow* window = app->window().glfwWindow();
        int winWidth, winHeight, fbWidth, fbHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        const float scaleX = static_cast<float>(fbWidth)  / static_cast<float>(winWidth);
        const float scaleY = static_cast<float>(fbHeight) / static_cast<float>(winHeight);
        const auto x = static_cast<uint32_t>(cursor.x * scaleX);
        const auto y = static_cast<uint32_t>(cursor.y * scaleY);
        app->pixelClicked(x, y, additive);
    }

    // Runs once per frame, after events are polled. Reads the accumulated
    // InputState instead of querying the backend directly.
    static void processInput(GLFWwindow* window, double deltaTime) {
        if (inputBlocked()) return;
        auto app = appOf(window);
        const InputState& input = app->input();
        const bool additive = input.isPressed(Key::LEFT_SHIFT) || input.isPressed(Key::RIGHT_SHIFT);

        if (input.wasPressed(Key::Q)) app->window().close();
        if (input.wasPressed(Key::Z)) app->toggleControlPanel();
        if (input.wasPressed(Key::ESCAPE)) app->clearSelection();
        if (input.wasPressed(Key::X)) objectSelected(app, additive);
        if (input.wasReleased(MouseButton::MIDDLE)) objectSelected(app, additive);

        auto camera = app->camera();
        const float speed = 0.06f * static_cast<float>(deltaTime);
        if (input.isPressed(Key::W)) { camera->move(speed, 0, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::S)) { camera->move(-speed, 0, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::A)) { camera->move(0, -speed, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::D)) { camera->move(0, speed, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::SPACE)) { camera->move(0, 0, -speed); app->onCameraMoved(); }
        if (input.isPressed(Key::LEFT_CONTROL)) { camera->move(0, 0, speed); app->onCameraMoved(); }

        const float rotateSpeed = speed * 0.3f;
        if (input.isPressed(Key::LEFT))  { camera->rotate(0, rotateSpeed, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::RIGHT)) { camera->rotate(0, -rotateSpeed, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::UP))    { camera->rotate(rotateSpeed, 0, 0); app->onCameraMoved(); }
        if (input.isPressed(Key::DOWN))  { camera->rotate(-rotateSpeed, 0, 0); app->onCameraMoved(); }

        if (input.isPressed(MouseButton::RIGHT)) {
            const glm::dvec2 delta = input.cursorDelta();
            if (delta.x != 0.0 || delta.y != 0.0) {
                constexpr double sensitivity = 0.1;
                camera->rotate(static_cast<float>(delta.y * sensitivity),
                               static_cast<float>(-delta.x * sensitivity), 0.f);
                app->onCameraMoved();
            }
        }

        const double scrollY = input.scrollDelta().y;
        if (scrollY != 0.0) {
            constexpr double zoomSpeed = 10.0;
            camera->zoom(static_cast<float>(scrollY * zoomSpeed));
            app->onCameraMoved();
        }
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
