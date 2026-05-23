//
// Created by igor on 4/18/26.
//

#ifndef COLLECTION_GPUTYPES_HPP
#define COLLECTION_GPUTYPES_HPP

#include "BVH.hpp"
#include "TLAS.hpp"
#include "Node.hpp"
#include "Types.hpp"

namespace crv::graphics::vulkan {
    using Triangle = PrecomputedTriangle<float>;
    using BLASNode = Node<float, 32, 3>;
    using TLASNode = Node<float, 32, 3>;
    using MeshPrimitive = MeshPrimitive<float>;
    using BLAS = BVH<BLASNode, Triangle>;
    using TLAS = BVH<TLASNode, MeshPrimitive>;
    using BLASBuilder = BinnedSAHBuilder<BLASNode, Triangle>;
    using TLASBuilder = BinnedSAHBuilder<TLASNode, MeshPrimitive>;

    struct alignas(16) TriangleGPU {
        glm::vec4 p0, e1, e2, N;
    };

    struct alignas(16) TriangleExtraGPU {
        glm::vec2 uv0, uv1, uv2;
        float padding[2];
    };

    struct alignas(16) BBoxGPU {
        glm::vec4 min, max;
    };

    struct alignas(16) NodeGPU {
        BBoxGPU bbox;
        uint32_t index;
        uint32_t pad[3];
    };

    struct alignas(16) CameraGPU {
        glm::vec4 pos;
        glm::mat4 invViewProj;
    };

    struct alignas(16) DirectLightGPU {
        glm::vec4 dir;
        float intensity;
        float pad[3];
    };

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

    struct alignas(16) MVPGPU {
        glm::mat4 model, view, proj, trInvModel;
    };

    struct alignas(16) RasterInstanceGPU {
        explicit RasterInstanceGPU(const MeshInstance& instance);
        glm::mat4 model;
        glm::mat4 invModel;
        uint32_t texIndex;
    };

    struct alignas(16) TracerInstanceGPU {
        explicit TracerInstanceGPU(const MeshInstance& instance);
        glm::mat4 model;
        glm::mat4 invModel;
        uint32_t baseNode;
        uint32_t baseTri; //can be removed
        uint32_t texIndex;
    };
}

#endif //COLLECTION_GPUTYPES_HPP