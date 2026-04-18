//
// Created by igor on 4/7/26.
//

#include "PathTracer.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "CallBacks.hpp"

namespace cvk = crv::graphics::vulkan;
namespace cg = crv::graphics;
namespace cs = crv::scene;
namespace cu = crv::utils;
namespace cm = crv::model;
using Tri = cg::PrecomputedTriangle<cvk::Scalar>;
using Vec4 = cvk::Vec4;

static int constexpr WIDTH  = 800;
static int constexpr HEIGHT = 600;

#define ROOM_PATH     ASSETS_PATH"room/scene.gltf"
#define LIMINAL_PATH  ASSETS_PATH"liminal/scene.gltf"
#define MICRO_PATH    ASSETS_PATH"microphone/scene.gltf"
#define DRAGON_PATH   ASSETS_PATH"DragonAttenuation/glTF/DragonAttenuation.gltf"
#define SWORD_PATH    ASSETS_PATH"Sword/sword.obj"

std::vector<Tri> loadModel(const glm::mat4& modelMatrix, const std::string& modelPath) {
    std::vector<Tri> triangles;
    cm::Loader loader;
    cu::Timer timer;
    timer.start();
    loader.setModel(modelPath);
    loader.load(modelMatrix);
    INFO << "Model load time: " << timer.duration() / 1000 << " sec";

    timer.start();
    const auto& indices = loader.indices();
    const auto& vertices = loader.vertices();
    const auto& meshes = loader.meshes();
    for (size_t i = 0; i < meshes.size(); ++i) {
        const auto& mesh = meshes[i];
        for (size_t j = 0; j < mesh.numIndices; j += 3) {
            const size_t idx = mesh.baseIndex + j;
            triangles.emplace_back(vertices[indices[idx + 0]].pos,
                                    vertices[indices[idx + 1]].pos,
                                    vertices[indices[idx + 2]].pos);
            //mMaterialIndices.emplace_back(mesh.materialIndex);
        }
    }
    INFO << "Primitive creation time: " << timer.duration() / 1000 << " sec";
    INFO << "Total number of primitives: " << triangles.size();
    //buildBVH(std::span(mPrimitives));
    return triangles;
}

void buildBVH(std::span<Tri> tris) {
    // cu::Timer timer;
    // timer.start();
    // cg::BinnedSAHBuilder<Node, Primitive> builder{ primitives };
    // mBvh = builder.build();
    // INFO << "BVH build time: " << timer.duration() / 1000 << " sec";
}

int main() {
    const cvk::WindowCreateInfo windowCreateInfo{
        .width = 800,
        .height = 600,
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
    model = glm::scale(model, glm::vec3(1));
    std::vector<cvk::AlignedTriangle> triangles;
    for (auto tri: loadModel(model, SWORD_PATH)) {
        triangles.emplace_back(Vec4(tri.p0, 1), Vec4(tri.e1, 1), Vec4(tri.e2, 1), Vec4(tri.N, 1));
    }
    const cvk::PathTracerCreateInfo pathTracerCreateInfo {
        .windowCreateInfo = windowCreateInfo,
        .cameraCreateInfo = cameraCreateInfo,
        .triangles = triangles
    };
    cvk::PathTracer pathTracer(pathTracerCreateInfo);
    setCallBacks(pathTracer.window(), pathTracer.camera());
    pathTracer.run();
}
