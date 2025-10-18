//
// Created by igor on 10/18/25.
//

#include "PathTracerApp.hpp"
#include "Sphere.hpp"

namespace graphics = crv::graphics;
namespace scene = crv::scene;
namespace app = crv::app;

using Sphere = graphics::Sphere<float>;
using Vec3 = Sphere::Vec3;

static int constexpr WIDTH  = 800;
static int constexpr HEIGHT = 600;

int main() {
    std::vector<graphics::Sphere<float>> spheres;
    spheres.emplace_back(Vec3(0, 50, 1000), 100);

    scene::CameraCreateInfo cameraCreateInfo {
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

    app::PathTracerAppCreateInfo appCreateInfo {
        .width = WIDTH,
        .height = HEIGHT,
        .spheres = spheres,
        .cameraCreateInfo = cameraCreateInfo
    };

    app::PathTracerApp app(appCreateInfo);
    app.run();
}