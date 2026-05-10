//
// Created by igor on 5/9/26.
//

#ifndef COLLECTION_BLAS_HPP
#define COLLECTION_BLAS_HPP
#include "BVH.hpp"

namespace crv::graphics {
    template <typename NodeType, typename Primitive>
    class BLAS: public BVH<NodeType, Primitive> {
    public:
        using Node = NodeType;
        using BvhType = BVH<Node, Primitive>;
        using Type = BvhType::Type;
        using Box = BBox<Type>;
        using Vec3 = Box::Vec3;
        using BvhType::mNodes;
        BLAS() = default;
        explicit BLAS(const BvhType& bvh): BvhType(bvh) {}
        BLAS& operator=(const BvhType& bvh) {
            BvhType::operator=(bvh);
            return *this;
        }
        [[nodiscard]] Box bbox() const {
            if (mNodes.empty()) return {};
            return mNodes[0].bbox();
        }
        [[nodiscard]] Vec3 center() const { return bbox().center(); }
    private:
    };
}



#endif //COLLECTION_BLAS_HPP