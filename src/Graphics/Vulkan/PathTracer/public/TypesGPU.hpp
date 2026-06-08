//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_TYPESGPU_HPP
#define COLLECTION_TYPESGPU_HPP

namespace crv::graphics::vulkan {
    struct PushConstants {

    };

    struct Vertex {
        glm::vec3 pos;
        glm::vec2 texCoord;
        glm::vec3 normal;
        glm::vec4 tangent;
    };

    struct alignas(16) CameraGPU {
        glm::mat4 invView;
        glm::mat4 invProj;
    };

    struct alignas(16) InstanceInfoGPU {
        uint meshID    = 0;
        uint textureID = 0;
        uint pad[2];
    };

    struct alignas(16) DirectLightGPU {
        glm::vec4 dir{};
        float intensity = 0;
        float pad[3];
    };
}

#endif //COLLECTION_TYPESGPU_HPP