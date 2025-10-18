//
// Created by igor on 10/17/25.
//

#include "Camera.hpp"
#include "Sphere.hpp"
#include "PathTracer.hpp"
#include "Window.hpp"
#include "GLUtils.hpp"

namespace graphics = crv::graphics;
namespace scene = crv::scene;

using Sphere = graphics::Sphere<float>;
using Vec3 = Sphere::Vec3;

static constexpr int WIDTH = 800;
static constexpr int HEIGHT = 600;


#include "PathTracerApp.hpp"

namespace crv::app {
    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo):
    mPathTracer(createInfo.spheres, createInfo.cameraCreateInfo),
    mWindow("Path Tracer", createInfo.width, createInfo.height) {}

    void PathTracerApp::run() {
        std::vector<uint8_t> imageBuffer = mPathTracer.render();
        mWindow.makeContextCurrent();

        if ( !initGLEW() ) return;
        const GLuint tex = createTexture(WIDTH, HEIGHT, imageBuffer.data());
        GLuint VAO, VBO, EBO;
        createBuffers(VAO, VBO, EBO);
        const GLuint shader = createShaderProgram();
        scene::AbsCamera* camera = mPathTracer.camera();

        while(!mWindow.shouldClose()) {
            camera->rotate( 0, 0, 2 );
            imageBuffer = mPathTracer.render();

            updateTexture(tex, WIDTH, HEIGHT, imageBuffer.data());

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
