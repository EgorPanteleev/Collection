//
// Created by igor on 10/17/25.
//

#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <stdexcept>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace crv::graphics {
    class Window {
    public:
        Window(const char* name, int width, int height): mWindow(nullptr) {
            initWindow(name, width, height);
        }

        ~Window() {
            if (mWindow) {
                glfwDestroyWindow(mWindow);
            }
            glfwTerminate();
        }

        bool shouldClose() const {
            return glfwWindowShouldClose(mWindow);
        }

        void makeContextCurrent() { glfwMakeContextCurrent(mWindow); }

        void getFrameBufferSize(int& width, int& height) { glfwGetFramebufferSize(mWindow, &width, &height); }

        void swapBuffers() { glfwSwapBuffers(mWindow); }
        static void pollEvents() { glfwPollEvents(); }
    private:
        void initWindow(const char* name, int width, int height) {
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

        GLFWwindow* mWindow;
    };
}

#endif //WINDOW_HPP
