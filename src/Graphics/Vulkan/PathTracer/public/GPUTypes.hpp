//
// Created by igor on 4/18/26.
//

#ifndef COLLECTION_GPUTYPES_HPP
#define COLLECTION_GPUTYPES_HPP

#include "BVH.hpp"
#include "Node.hpp"
#include "Texture.hpp"

namespace crv::graphics::vulkan {
    using Scalar = float;
    using Vec2 = glm::vec<2, Scalar>;
    using Vec3 = glm::vec<3, Scalar>;
    using Vec4 = glm::vec<4, Scalar>;
    using Tri = PrecomputedTriangle<Scalar>;
    using BVH = BVH<Node<Scalar, 32, 4>, Tri>;
    struct alignas(16) AlignedTriangle {
        Vec4 p0, e1, e2, N;
    };

    struct alignas(16) AlignedTriangleExtra {
        Vec2 uv0, uv1, uv2;
        Vec2 padding;
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
        Vec4 position, forward, right, up;
        float FOV, aspectRatio, nearPlane, farPlane;
    };

    struct alignas(16) AlignedDirectLight {
        Vec4 dir;
        float intensity;
        float pad[3];
    };

    struct PushConstants {
        uint32_t frame, maxDepth;
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
        glm::mat4 model, view, proj;
    };

    struct GBuffer {
        Image     colorImage{};
        ImageView colorView{};
        Image     depthImage{};
        ImageView depthView{};
        Sampler   sampler{};
    };
}

#endif //COLLECTION_GPUTYPES_HPP