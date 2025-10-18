//
// Created by igor on 10/17/25.
//

#include "PathTracer.hpp"
#include "Window.hpp"
#include "GLUtils.hpp"
#include "PathTracerApp.hpp"
#include "Timer.hpp"

namespace crv::app {
    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo):
    mPathTracer(createInfo.spheres, createInfo.cameraCreateInfo),
    mWindow(title(), createInfo.width, createInfo.height) {}

    void PathTracerApp::run() {
        std::vector<uint8_t> imageBuffer = mPathTracer.render();
        mWindow.makeContextCurrent();

        if ( !initGLEW() ) return;
        const GLuint tex = createTexture(mWindow.width(), mWindow.height(), imageBuffer.data());
        GLuint VAO, VBO, EBO;
        createBuffers(VAO, VBO, EBO);
        const GLuint shader = createShaderProgram();
        scene::AbsCamera* camera = mPathTracer.camera();
        utils::FpsCounter fpsCounter;

        while(!mWindow.shouldClose()) {
            fpsCounter.update();
            std::string newTitle(title());
            mWindow.setTitle( (newTitle + "(" + fpsCounter.fpsAsString() + " fps)").c_str() );
            camera->rotate( 0, 0, 2 );
            imageBuffer = mPathTracer.render();

            updateTexture(tex, mWindow.width(), mWindow.height(), imageBuffer.data());

            int winWidth, winHeight;
            mWindow.getFrameBufferSize(winWidth, winHeight);
            glViewport(0, 0, winWidth, winHeight);

            drawTexture(shader, VAO, tex);

            mWindow.swapBuffers();
            graphics::Window::pollEvents();
        }

        cleanData(tex, shader, VBO, EBO, VAO);
    }
}
