//
// Created by igor on 4/7/26.
//

#ifndef COLLECTION_BASEWINDOW_HPP
#define COLLECTION_BASEWINDOW_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>

namespace crv::graphics {
    class BaseWindow {
    public:
        BaseWindow() = default;
        BaseWindow(const char* name, int width, int height);
        BaseWindow(const BaseWindow&) = delete;
        BaseWindow& operator=(const BaseWindow&) = delete;
        BaseWindow(BaseWindow&&) noexcept;
        BaseWindow& operator=(BaseWindow&&) noexcept;

        virtual ~BaseWindow();
        [[nodiscard]] bool shouldClose() const { return glfwWindowShouldClose(mWindow); }
        void makeContextCurrent() const { glfwMakeContextCurrent(mWindow); }
        void getFrameBufferSize(int& width, int& height) const { glfwGetFramebufferSize(mWindow, &width, &height); }
        void swapBuffers() const { glfwSwapBuffers(mWindow); }
        void setTitle(const char* title) const { glfwSetWindowTitle(mWindow, title); }
        void setKeyCallBack(GLFWkeyfun callback) const { glfwSetKeyCallback(mWindow, callback); }
        void setMouseButtonCallBack(GLFWmousebuttonfun callback) const { glfwSetMouseButtonCallback(mWindow, callback); }
        void setMouseMoveCallBack(GLFWcursorposfun callback) const { glfwSetCursorPosCallback(mWindow, callback); }
        void setScrollCallBack(GLFWscrollfun callback) const { glfwSetScrollCallback(mWindow, callback); }
        void setUserPoint(void* pointer) { glfwSetWindowUserPointer(mWindow, pointer); }
        [[nodiscard]] void* getUserPoint() const { return glfwGetWindowUserPointer(mWindow); }
        static void pollEvents() { glfwPollEvents(); }
        [[nodiscard]] int width() const { return mWidth; }
        [[nodiscard]] int height() const { return mHeight; }
        [[nodiscard]] GLFWwindow* glfwWindow() const { return mWindow; }
        void close() const { glfwSetWindowShouldClose(mWindow, GLFW_TRUE); }
        virtual void init() = 0;
    protected:
        GLFWwindow* mWindow = nullptr;
        int mWidth = 800;
        int mHeight = 600;
        std::string mName = "BaseWindow";
    };
}

#endif //COLLECTION_BASEWINDOW_HPP