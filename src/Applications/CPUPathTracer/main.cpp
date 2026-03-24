//
// Created by igor on 10/18/25.
//

#include "PathTracerApp.hpp"
#include "Loader.hpp"
#include "Message.hpp"
#include "SweepSAHBuilder.hpp"
#include "Node.hpp"
#include "Timer.hpp"

namespace cg = crv::graphics;
namespace cs = crv::scene;
namespace capp = crv::app;
namespace cm = crv::model;
namespace cu = crv::utils;

using Type = float;

static int constexpr WIDTH  = 800;
static int constexpr HEIGHT = 600;

#define ASSETS_PATH "/home/igor/dev/src/Collection/assets/"

#define ROOM_PATH     ASSETS_PATH"room/scene.gltf"
#define LIMINAL_PATH  ASSETS_PATH"liminal/scene.gltf"
#define MICRO_PATH    ASSETS_PATH"microphone/scene.gltf"
#define DRAGON_PATH   ASSETS_PATH"DragonAttenuation/glTF/DragonAttenuation.gltf"

int main() {
    cs::CameraCreateInfo cameraCreateInfo {
        .type = cs::CameraType::FLY,
        .pos = glm::vec3(50, 10, 0),
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
    model = glm::scale(model, glm::vec3(100));
    capp::PathTracerAppCreateInfo appCreateInfo {
        .cameraCreateInfo = cameraCreateInfo,
        .model = model,
        .modelPath = ROOM_PATH,
        .width = WIDTH,
        .height = HEIGHT
    };

    capp::PathTracerApp<Type, 4> app(appCreateInfo);
    app.run();


    // ---- ROOM ------
    // BVH build (Sweep SAH) - 2.94
    // Fps - 30

    // BVH build (Sweep SAH) - 0.021
    // Fps - 28
}