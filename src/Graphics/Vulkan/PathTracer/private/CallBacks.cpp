//
// Created by igor on 4/18/26.
//

#include "CallBacks.hpp"

static bool rightMouseButtonPressed = false;
static double lastX = 0.0f, lastY = 0.0f;

namespace crv::graphics::vulkan {
    static void processKeyboard(GLFWwindow* window, double deltaTime) {
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        auto camera = app->camera();
        const float speed = 0.06f * deltaTime;
        //if (speed < 0) return;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera->move(speed, 0, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera->move(-speed, 0, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera->move(0, -speed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera->move(0, speed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            camera->move(0, 0, -speed);
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            camera->move(0, 0, speed);
        }
        float rotateSpeed = speed * 0.3f;

        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            camera->rotate(0, rotateSpeed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            camera->rotate(0, -rotateSpeed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            camera->rotate(rotateSpeed, 0, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            camera->rotate(-rotateSpeed, 0, 0);
        }
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        auto camera = static_cast<cs::AbsCamera*>(glfwGetWindowUserPointer(window));
        float speed = 10.0f;
        camera->zoom(yoffset * speed);
    }

    static void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto app = static_cast<PathTracerApp*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (action == GLFW_PRESS && key == GLFW_KEY_Z) {
            app->toggleControlPanel();
        }
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                rightMouseButtonPressed = true;
                glfwGetCursorPos(window, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                rightMouseButtonPressed = false;
            }
        }
    }

    void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
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