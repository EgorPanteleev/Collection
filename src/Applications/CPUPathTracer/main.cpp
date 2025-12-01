//
// Created by igor on 10/18/25.
//

#include "PathTracerApp.hpp"
#include "Loader.hpp"
#include "Message.hpp"
#include "AbsBuilder.hpp"
#include "Node.hpp"

namespace graphics = crv::graphics;
namespace scene = crv::scene;
namespace app = crv::app;
namespace model = crv::model;

using PreTri = graphics::PrecomputedTriangle<float>;
using Vec3 = PreTri::Vec3;

static int constexpr WIDTH  = 800;
static int constexpr HEIGHT = 600;

int main() {
    model::Loader loader("/home/igor/dev/src/Collection/assets/DragonAttenuation/glTF/DragonAttenuation.gltf");
    loader.load();
    std::vector<PreTri> triangles;
    const auto& vertices = loader.vertices();
    for (int i = 0; i < loader.indices().size(); i+=3) {
        uint32_t idx = loader.indices()[i];
        triangles.emplace_back(vertices[idx].pos, vertices[idx+1].pos, vertices[idx+2].pos);
    }

    graphics::AbsBuilder<graphics::Node<float>> builder(graphics::AbsBuilder<graphics::Node<float>>::BINNED_SAH);
    builder.build(std::span(triangles.data(), triangles.size()));

    MESSAGE << "Builded..";
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
        .triangles = triangles,
        .cameraCreateInfo = cameraCreateInfo
    };

    app::PathTracerApp app(appCreateInfo);
    app.run();
}