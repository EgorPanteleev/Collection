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

int main() {
    std::vector<Sphere> spheres;
    spheres.emplace_back(Vec3(0, 5, 100), 10);

    scene::CameraCreateInfo cameraCreateInfo{
        .type = scene::CameraType::FLY,
        .pos = glm::vec3(0),
        .target = glm::vec3(0, 0, 1),
        .up = glm::vec3(0, 1, 0),
        .zoom = 1,
        .FOV = 60,
        .aspectRatio = static_cast<float>(WIDTH) / HEIGHT,
        .nearPlane = 0,
        .farPlane = 100000.0f,
    };
    graphics::PathTracer pathTracer(spheres, cameraCreateInfo);
    std::vector<uint8_t> imageBuffer = pathTracer.render();

    graphics::Window window("CPU path tracer", WIDTH, HEIGHT);

    window.makeContextCurrent();

    if ( !initGLEW() ) return -1;
    GLuint tex = createTexture(WIDTH, HEIGHT, imageBuffer.data());
    GLuint VAO, VBO, EBO;
    createBuffers(VAO, VBO, EBO);
    GLuint shader = createShaderProgram();
    scene::AbsCamera* camera = pathTracer.camera();

    while(!window.shouldClose()) {
        camera->rotate( 0, 0, 2 );
        imageBuffer = pathTracer.render();

        updateTexture(tex, WIDTH, HEIGHT, imageBuffer.data());

        int winWidth, winHeight;
        window.getFrameBufferSize(winWidth, winHeight);
        glViewport(0, 0, winWidth, winHeight);

        drawTexture(shader, VAO, tex);

        window.swapBuffers();
        graphics::Window::pollEvents();
    }

    cleanData(tex, shader, VBO, EBO, VAO);
}
