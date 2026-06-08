//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_TYPESGPU_HPP
#define COLLECTION_TYPESGPU_HPP

namespace crv::graphics::vulkan {
    struct PushConstants {
        uint32_t frameCount  = 0;
        uint32_t spp         = 1;
        uint32_t minDepth    = 1;
        uint32_t maxDepth    = 1;
        uint32_t displayMode = 4;
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