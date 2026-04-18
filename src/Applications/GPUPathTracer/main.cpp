//
// Created by igor on 4/7/26.
//

#include "PathTracer.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "CallBacks.hpp"
#include "BinnedSAHBuilder.hpp"

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

cvk::BVH buildBVH(std::span<Tri> tris) {
    cu::Timer timer;
    timer.start();
    cg::BinnedSAHBuilder<cvk::BVH::Node, Tri> builder{ tris };
    cvk::BVH bvh = builder.build();
    INFO << "BVH build time: " << timer.duration() / 1000 << " sec";
    return bvh;
}

std::tuple<std::vector<Tri>, cvk::BVH> loadModel(const glm::mat4& modelMatrix, const std::string& modelPath) {
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
    return {triangles, buildBVH(std::span(triangles))};
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
    model = glm::scale(model, glm::vec3(100));
    std::vector<cvk::AlignedTriangle> triangles;
    auto [tris, bvh] =  loadModel(model, DRAGON_PATH);
    for (const auto& tri: tris) {
        triangles.emplace_back(Vec4(tri.p0, 1), Vec4(tri.e1, 1), Vec4(tri.e2, 1), Vec4(tri.N, 1));
    }
    std::vector<cvk::AlignedNode> nodes;
    for (const auto& node: bvh.nodes()) {
        cvk::AlignedNode alignedNode{};
        alignedNode.bbox = cvk::AlignedBBox(Vec4(node.bbox().min, 1), Vec4(node.bbox().max, 1));
        alignedNode.index = node.index().value();
        nodes.push_back(alignedNode);
    }
    const cvk::PathTracerCreateInfo pathTracerCreateInfo {
        .windowCreateInfo = windowCreateInfo,
        .cameraCreateInfo = cameraCreateInfo,
        .triangles = triangles,
        .nodes = nodes,
        .indexes = bvh.primIds()
    };
    cvk::PathTracer pathTracer(pathTracerCreateInfo);
    setCallBacks(pathTracer.window(), pathTracer.camera());
    pathTracer.run();
}
