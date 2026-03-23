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
using PreTri = cg::PrecomputedTriangle<Type>;
using Vec3 = PreTri::Vec3;
using NodeType = cg::Node<Type>;

static int constexpr WIDTH  = 800;
static int constexpr HEIGHT = 600;

#define YEY_PATH "/home/igor/dev/src/Collection/assets/eye/obj/eyeball.obj"
#define SWORD_PATH "/home/igor/dev/src/Collection/assets/sword/sword.obj"
#define DRAGON_PATH "/home/igor/dev/src/Collection/assets/DragonAttenuation/glTF/DragonAttenuation.gltf"

int main() {
    cu::Timer timer;
    timer.start();
    cm::Loader loader(SWORD_PATH);
    loader.load();
    INFO << "Load time: " << timer.duration() / 1000 << " sec";
    std::vector<PreTri> triangles;
    const auto& indices = loader.indices();
    const auto& vertices = loader.vertices();
    for (int i = 0; i < indices.size(); i += 3) {
        triangles.emplace_back(vertices[indices[i]].pos, vertices[indices[i + 1]].pos, vertices[indices[i + 2]].pos);
    }

    timer.start();
    cg::SweepSAHBuilder<NodeType, PreTri> builder{ std::span(triangles) };
    auto bvh = builder.build();
    INFO << "Build time: " << timer.duration() / 1000 << " sec";
    cs::CameraCreateInfo cameraCreateInfo {
        .type = cs::CameraType::FLY,
        .pos = glm::vec3(0, 0, -5),
        .target = glm::vec3(0, 0, 1),
        .up = glm::vec3(0, 1, 0),
        .zoom = 1,
        .FOV = 60,
        .aspectRatio = static_cast<float>(WIDTH) / HEIGHT,
        .nearPlane = 0,
        .farPlane = 100000.0f,
    };

    capp::PathTracerAppCreateInfo<NodeType, PreTri> appCreateInfo {
        .width = WIDTH,
        .height = HEIGHT,
        .bvh = bvh,
        .cameraCreateInfo = cameraCreateInfo
    };

    capp::PathTracerApp app(appCreateInfo);
    app.run();
}