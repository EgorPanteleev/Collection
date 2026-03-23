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

#define MICRO_PATH "/home/igor/dev/src/Collection/assets/microphone/scene.gltf"
#define EYE_PATH "/home/igor/dev/src/Collection/assets/eye/obj/eyeball.obj"
#define SWORD_PATH "/home/igor/dev/src/Collection/assets/sword/sword.obj"
#define DRAGON_PATH "/home/igor/dev/src/Collection/assets/DragonAttenuation/glTF/DragonAttenuation.gltf"

int main() {
    cs::CameraCreateInfo cameraCreateInfo {
        .type = cs::CameraType::FLY,
        .pos = glm::vec3(0, 0, -5),
        .target = glm::vec3(0, 0, 1),
        .up = glm::vec3(0, 1, 0),
        .zoom = 1,
        .FOV = 60,
        .aspectRatio = static_cast<float>(WIDTH) / HEIGHT,
        .nearPlane = 0,
        .farPlane = 500.0f,
    };

    capp::PathTracerAppCreateInfo appCreateInfo {
        .cameraCreateInfo = cameraCreateInfo,
        .modelPath = MICRO_PATH,
        .width = WIDTH,
        .height = HEIGHT
    };

    capp::PathTracerApp<Type, 4> app(appCreateInfo);
    app.run();
}