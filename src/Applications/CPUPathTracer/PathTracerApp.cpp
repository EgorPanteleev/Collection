//
// Created by igor on 10/17/25.
//

#include "PathTracer.hpp"
#include "Window.hpp"
#include "GLUtils.hpp"
#include "PathTracerApp.hpp"
#include "Timer.hpp"
#include <iostream>

namespace crv::app {
    bool rightMouseButtonPressed = false;
    double lastX = 0.0f, lastY = 0.0f;

    static void processKeyboard(GLFWwindow* window, crv::scene::AbsCamera* camera, double deltaTime) {
        auto speed = static_cast<float>(deltaTime);
        if (speed < 0) return;
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

    static void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto app = static_cast<crv::app::PathTracerApp*>(glfwGetWindowUserPointer(window));

        if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
            app->quit();
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
        auto app = static_cast<crv::app::PathTracerApp*>(glfwGetWindowUserPointer(window));
        crv::scene::AbsCamera* camera = app->mPathTracer.camera();

        if (!rightMouseButtonPressed || !camera) return;

        double sensitivity = 0.1f;
        double offsetX = xpos - lastX;
        double offsetY = lastY - ypos;

        lastX = xpos;
        lastY = ypos;
        camera->rotate((float)(-offsetY * sensitivity), (float)(-offsetX * sensitivity), 0.f);
    }

    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo):
    mPathTracer(createInfo.triangles, createInfo.cameraCreateInfo),
    mWindow(title(), createInfo.width, createInfo.height) {}

    void PathTracerApp::run() {
        std::vector<uint8_t> imageBuffer;
        mWindow.makeContextCurrent();
        mWindow.setUserPoint(this);
        mWindow.setKeyCallBack(keyCallBack);
        mWindow.setMouseButtonCallBack(mouseButtonCallback);
        mWindow.setMouseMoveCallBack(mouseMoveCallback);

        if ( !initGLEW() ) return;
        const GLuint tex = createTexture(mWindow.width(), mWindow.height(), imageBuffer.data());
        GLuint VAO, VBO, EBO;
        createBuffers(VAO, VBO, EBO);
        const GLuint shader = createShaderProgram();
        scene::AbsCamera* camera = mPathTracer.camera();
        utils::FpsCounter fpsCounter;
        double deltaTime = 0;
        while(!mWindow.shouldClose()) {
            fpsCounter.update();
            std::string newTitle(title());
            mWindow.setTitle( (newTitle + "(" + fpsCounter.fpsAsString() + " fps)").c_str() );
            deltaTime = 1e3 / fpsCounter.fps();
            imageBuffer = mPathTracer.render();

            updateTexture(tex, mWindow.width(), mWindow.height(), imageBuffer.data());

            int winWidth, winHeight;
            mWindow.getFrameBufferSize(winWidth, winHeight);
            glViewport(0, 0, winWidth, winHeight);

            drawTexture(shader, VAO, tex);

            mWindow.swapBuffers();
            graphics::Window::pollEvents();
            processKeyboard(mWindow.glfwWindow(), camera, deltaTime);
        }
        cleanData(tex, shader, VBO, EBO, VAO);
    }
}
