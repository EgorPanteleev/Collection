//
// Created by igor on 4/18/26.
//

#ifndef COLLECTION_GPUTYPES_HPP
#define COLLECTION_GPUTYPES_HPP

#include "BVH.hpp"
#include "BLAS.hpp"
#include "TLAS.hpp"
#include "Node.hpp"
#include "Texture.hpp"

namespace crv::graphics::vulkan {
    using Scalar = float;
    using Vec2 = glm::vec<2, Scalar>;
    using Vec3 = glm::vec<3, Scalar>;
    using Vec4 = glm::vec<4, Scalar>;
    using Tri = PrecomputedTriangle<Scalar>;
    using Node = Node<Scalar, 32, 3>;
    using BVH = BVH<Node, Tri>;
    using BLAS = BLAS<Node, Tri>;
    using TLAS = TLAS<Node, Tri>;
    using MeshPrimitive = MeshPrimitive<float>;
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
        uint32_t frame, spp, minDepth, maxDepth, instanceCount;
    };

    struct TexturesByType {
        Texture& operator[](const int type) { return mTexturesByType[type]; }
        std::array<Texture, cm::Texture::UNKNOWN> mTexturesByType{};
    };

    struct Vertex {
        Vec3 pos;
        Vec2 texCoord;
        Vec3 normal;
        Vec4 tangent;
        uint32_t texIndex;
    };

    struct alignas(16) AlignedMVP {
        glm::mat4 model, view, proj, trInvModel;
    };

    struct alignas(16) AlignedInstance {
        glm::mat4 model;
    };

    struct GBuffer {
        Image     colorImage{};
        ImageView colorView{};
        Image     depthImage{};
        ImageView depthView{};
        Image     normalImage{};
        ImageView normalView{};
        Sampler   sampler{};
    };

    struct alignas(16) MeshInstance {
        glm::mat4 model;
        glm::mat4 invModel;
        uint32_t baseNode;
        uint32_t baseTri;
    };

    struct MeshData {
        std::vector<Vertex>          vertices{};
        std::vector<uint32_t>        indices{};
        std::vector<MeshInstance>    instances{};
    };
}

#endif //COLLECTION_GPUTYPES_HPP