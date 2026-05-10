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

#define ROOM_PATH     SCENES_PATH"room/scene.gltf"
#define LIMINAL_PATH  SCENES_PATH"liminal/scene.gltf"
#define MICRO_PATH    SCENES_PATH"microphone/scene.gltf"
#define DRAGON_PATH   MESHES_PATH"dragon.glb"
#define SPHERE_PATH   MESHES_PATH"sphere.glb"
#define SWORD_PATH    SCENES_PATH"Sword/sword.obj"
#define SPONZA_PATH   SCENES_PATH"Sponza/glTF/Sponza.gltf"
#define FORD_PATH     MESHES_PATH"ford.obj"

int main() {
    cvk::PathTracerAppCreateInfo createInfo {
        .scenePath = ASSETS_PATH"scene.json",
        .directLight = cvk::AlignedDirectLight(Vec4(0.132, 0.066, 0.970, 1), 2.0)
    };
    cvk::PathTracerApp app(createInfo);
    app.run();
}