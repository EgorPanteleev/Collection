//
// Created by igor on 4/18/26.
//

#ifndef COLLECTION_GPUTYPES_HPP
#define COLLECTION_GPUTYPES_HPP

#include "BVH.hpp"
#include "TLAS.hpp"
#include "Node.hpp"
#include "Texture.hpp"

namespace crv::graphics::vulkan {
    using Scalar = float;
    using Vec2 = glm::vec<2, Scalar>;
    using Vec3 = glm::vec<3, Scalar>;
    using Vec4 = glm::vec<4, Scalar>;
    using Tri = PrecomputedTriangle<Scalar>;
    using BLASNode = Node<Scalar, 32, 3>;
    using TLASNode = Node<Scalar, 32, 3>;
    using MeshPrimitive = MeshPrimitive<float>;
    using BLAS = BVH<BLASNode, Tri>;
    using TLAS = BVH<TLASNode, MeshPrimitive>;
    using BLASBuilder = BinnedSAHBuilder<BLASNode, Tri>;
    using TLASBuilder = BinnedSAHBuilder<TLASNode, MeshPrimitive>;

    struct alignas(16) AlignedTriangle {
        Vec4 p0, e1, e2, N;
    };

    struct alignas(16) AlignedTriangleExtra {
        Vec2 uv0, uv1, uv2;
        uint32_t texIndex;
        float padding;
    };

    struct alignas(16) AlignedBBox {
        Vec4 min, max;
    };

    struct alignas(16) AlignedNode {
        AlignedBBox bbox;
        uint32_t index;
        uint32_t pad[3];
    };

    struct alignas(16) AlignedCamera {
        Vec4 pos;
        glm::mat4 invViewProj;
    };

    struct alignas(16) AlignedDirectLight {
        Vec4 dir;
        float intensity;
        float pad[3];
    };

    struct PushConstants {
        uint32_t frame, spp, minDepth, maxDepth;
    };

    struct Vertex {
        Vec3 pos;
        Vec2 texCoord;
        Vec3 normal;
        Vec4 tangent;
    };

    struct alignas(16) AlignedMVP {
        glm::mat4 model, view, proj, trInvModel;
    };

    struct MeshInstance {
        std::string name;
        glm::mat4 model;
        glm::mat4 invModel;
        uint32_t baseNode;
        uint32_t baseTri; //can be removed
        uint32_t texIndex;
    };

    struct alignas(16) RasterInstance {
        explicit RasterInstance(const MeshInstance& instance): model(instance.model), invModel(instance.invModel),
        texIndex(instance.texIndex) {}
        glm::mat4 model;
        glm::mat4 invModel;
        uint32_t texIndex;
    };

    struct alignas(16) TracerInstance {
        explicit TracerInstance(const MeshInstance& instance): model(instance.model), invModel(instance.invModel),
        baseNode(instance.baseNode), baseTri(instance.baseTri), texIndex(instance.texIndex) {}
        glm::mat4 model;
        glm::mat4 invModel;
        uint32_t baseNode;
        uint32_t baseTri; //can be removed
        uint32_t texIndex;
    };
}

#endif //COLLECTION_GPUTYPES_HPP