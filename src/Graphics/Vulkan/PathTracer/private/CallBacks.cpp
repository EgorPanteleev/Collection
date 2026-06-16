//
// Created by igor on 6/8/26.
//

#include "CallBacks.hpp"

#include <imgui_internal.h>

static bool rightMouseButtonPressed = false;
static double lastX = 0.0f, lastY = 0.0f;

namespace crv::graphics::vulkan {
    static bool inputBlocked() {
        return ImGui::GetTopMostPopupModal() != nullptr;
    }

    static void processKeyboard(GLFWwindow* window, double deltaTime) {
        if (inputBlocked()) return;
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        auto camera = app->camera();
        const float speed = 0.06f * deltaTime;
        //if (speed < 0) return;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera->move(speed, 0, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera->move(-speed, 0, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera->move(0, -speed, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera->move(0, speed, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            camera->move(0, 0, -speed);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            camera->move(0, 0, speed);
            app->updateImage();
        }
        float rotateSpeed = speed * 0.3f;

        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            camera->rotate(0, rotateSpeed, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            camera->rotate(0, -rotateSpeed, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            camera->rotate(rotateSpeed, 0, 0);
            app->updateImage();
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            camera->rotate(-rotateSpeed, 0, 0);
            app->updateImage();
        }
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        if (inputBlocked()) return;
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        auto camera = app->camera();
        float speed = 10.0f;
        camera->zoom(yoffset * speed);
        app->updateImage();
    }

    static void objectSelected(GLFWwindow* window) {
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        int winWidth, winHeight;
        int fbWidth, fbHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        float scaleX = static_cast<float>(fbWidth)  / static_cast<float>(winWidth);
        float scaleY = static_cast<float>(fbHeight) / static_cast<float>(winHeight);
        auto x = static_cast<uint32_t>(mouseX * scaleX);
        auto y = static_cast<uint32_t>(mouseY * scaleY);
        app->pixelClicked(x, y);
    }

    static void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (inputBlocked()) return;
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (action == GLFW_PRESS && key == GLFW_KEY_Z) {
            app->toggleControlPanel();
        }
        if (action == GLFW_PRESS && key == GLFW_KEY_X) {
            objectSelected(window);
        }
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (inputBlocked()) return;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                rightMouseButtonPressed = true;
                glfwGetCursorPos(window, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                rightMouseButtonPressed = false;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_RELEASE) {
                objectSelected(window);
            }
        }
    }

    void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
        if (inputBlocked()) return;
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        auto camera = app->camera();
        if (!rightMouseButtonPressed || !camera) return;

        double sensitivity = 0.1f;
        double offsetX = xpos - lastX;
        double offsetY = lastY - ypos;

        lastX = xpos;
        lastY = ypos;
        camera->rotate(static_cast<float>(-offsetY * sensitivity),
                       static_cast<float>(-offsetX * sensitivity), 0.f);
        app->updateImage();
    }

    void setCallBacks(PathTracerApp* app) {
        Window& window = app->window();
        window.setUserPoint(app);
        window.makeContextCurrent();
        window.setKeyboardCallBack(processKeyboard);
        window.setKeyCallBack(keyCallBack);
        window.setMouseButtonCallBack(mouseButtonCallback);
        window.setMouseMoveCallBack(mouseMoveCallback);
        window.setScrollCallBack(scrollCallback);
    }
}