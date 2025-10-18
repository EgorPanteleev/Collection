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
        Window(const char* name, int width, int height);

        ~Window();
        bool shouldClose() const { return glfwWindowShouldClose(mWindow); }
        void makeContextCurrent() { glfwMakeContextCurrent(mWindow); }
        void getFrameBufferSize(int& width, int& height) { glfwGetFramebufferSize(mWindow, &width, &height); }
        void swapBuffers() { glfwSwapBuffers(mWindow); }
        static void pollEvents() { glfwPollEvents(); }
        int width() { return mWidth; }
        int height() { return mHeight; }
    private:
        void initWindow(const char* name);

        GLFWwindow* mWindow;
        int mWidth;
        int mHeight;
    };
}

#endif //WINDOW_HPP
