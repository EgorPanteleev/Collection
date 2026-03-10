//
// Created by igor on 10/17/25.
//

#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace crv::graphics {
    class Window {
    public:
        Window(const char* name, int width, int height);

        ~Window();
        [[nodiscard]] bool shouldClose() const { return glfwWindowShouldClose(mWindow); }
        void makeContextCurrent() const { glfwMakeContextCurrent(mWindow); }
        void getFrameBufferSize(int& width, int& height) const { glfwGetFramebufferSize(mWindow, &width, &height); }
        void swapBuffers() const { glfwSwapBuffers(mWindow); }
        void setTitle(const char* title) const { glfwSetWindowTitle(mWindow, title); }
        void setKeyCallBack(GLFWkeyfun callback) const { glfwSetKeyCallback(mWindow, callback); }
        void setMouseButtonCallBack(GLFWmousebuttonfun callback) const { glfwSetMouseButtonCallback(mWindow, callback); }
        void setMouseMoveCallBack(GLFWcursorposfun callback) const { glfwSetCursorPosCallback(mWindow, callback); }
        void setUserPoint(void* pointer) { glfwSetWindowUserPointer(mWindow, pointer); }
        [[nodiscard]] void* getUserPoint() const { return glfwGetWindowUserPointer(mWindow); }
        static void pollEvents() { glfwPollEvents(); }
        [[nodiscard]] int width() const { return mWidth; }
        [[nodiscard]] int height() const { return mHeight; }
        [[nodiscard]] GLFWwindow* glfwWindow() const { return mWindow; }
        void close() const { glfwSetWindowShouldClose(mWindow, GLFW_TRUE); }
    private:
        void initWindow(const char* name);

        GLFWwindow* mWindow;
        int mWidth;
        int mHeight;
    };
}

#endif //WINDOW_HPP
