//
// Created by igor on 4/22/26.
//

#include "PathTracerApp.hpp"

namespace cvk = crv::graphics::vulkan;
namespace cg = crv::graphics;
namespace cs = crv::scene;
namespace cu = crv::utils;
namespace cm = crv::model;
using Tri = cg::PrecomputedTriangle<cvk::Scalar>;
using Vec2 = cvk::Vec2;
using Vec3 = cvk::Vec3;
using Vec4 = cvk::Vec4;

static int constexpr WIDTH  = 800;
static int constexpr HEIGHT = 600;

#define ROOM_PATH     ASSETS_PATH"room/scene.gltf"
#define LIMINAL_PATH  ASSETS_PATH"liminal/scene.gltf"
#define MICRO_PATH    ASSETS_PATH"microphone/scene.gltf"
#define DRAGON_PATH   ASSETS_PATH"DragonAttenuation/glTF/DragonAttenuation.gltf"
#define SWORD_PATH    ASSETS_PATH"Sword/sword.obj"
#define SPONZA_PATH   ASSETS_PATH"Sponza/glTF/Sponza.gltf"

int main() {
    const cvk::WindowCreateInfo windowCreateInfo{
        .width = WIDTH,
        .height = HEIGHT,
        .name = "GPU Path Tracer"
    };
    const cs::CameraCreateInfo cameraCreateInfo{
        .type = cs::CameraType::FLY,
        .pos = glm::vec3(80, 10, 0),
        .target = glm::vec3(-100, 10, 0),
        .up = glm::vec3(0, 1, 0),
        .zoom = 1,
        .FOV = 60,
        .aspectRatio = static_cast<float>(WIDTH) / HEIGHT,
        .nearPlane = 0,
        .farPlane = 5000.0f,
    };
    auto model = glm::mat4(1.);
    // model = glm::translate(model, glm::vec3(0, 0, 0));
    model = glm::rotate(model, glm::radians(180.f), glm::vec3(1, 0, 0));
    model = glm::scale(model, glm::vec3(50));

    cvk::PathTracerAppCreateInfo createInfo {
        .windowCreateInfo = windowCreateInfo,
        .cameraCreateInfo = cameraCreateInfo,
        .modelMatrix = model,
        .modelPath = ROOM_PATH,
        .directLight = cvk::AlignedDirectLight(Vec4(1, -0.2, -0.3, 1), 0.7)
    };
    cvk::PathTracerApp app(createInfo);
    app.run();
}