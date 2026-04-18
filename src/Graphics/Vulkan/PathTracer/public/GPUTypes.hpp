//
// Created by igor on 4/18/26.
//

#ifndef COLLECTION_GPUTYPES_HPP
#define COLLECTION_GPUTYPES_HPP

#include "BVH.hpp"
#include "Node.hpp"

namespace crv::graphics::vulkan {
    using Scalar = float;
    using Vec4 = glm::vec<4, Scalar>;
    using Tri = PrecomputedTriangle<Scalar>;
    using BVH = BVH<Node<Scalar, 32, 4>, Tri>;
    struct alignas(16) AlignedTriangle {
        Vec4 p0, e1, e2, N;
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

    struct PushConstants {
        int width;
        int height;
    };
}

#endif //COLLECTION_GPUTYPES_HPP