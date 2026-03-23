//
// Created by igor on 3/10/26.
//

#ifndef COLLECTION_BVH_HPP
#define COLLECTION_BVH_HPP

#include <vector>
#include <stack>
#include <optional>

#include "Ray.hpp"
#include "Triangle.hpp"

namespace crv::graphics {
    template <typename Node, typename Primitive>
    class SweepSAHBuilder;

    template <typename N, typename Primitive>
    class BVH {
    public:
        using Node = N;
        using Type = Node::Type;
        using IndexType = Node::IndexType;
        friend class SweepSAHBuilder<Node, Primitive>;

        std::optional<std::tuple<Type, Type, Type>> intersect(const Ray<Type>& ray, Type eps) const {
            std::optional<std::tuple<Type, Type, Type>> closestHit;
            Type closestT = std::numeric_limits<Type>::max();

            // for (auto& prim : mPrimitives) {
            //     auto hit = prim.intersect(ray, 0);
            //
            // if (hit) {
            //     auto [t, u, v] = *hit;
            //     if (t < closestT) {
            //         closestT = t;
            //         closestHit = *hit;
            //     }
            // }
            // }
            // return closestHit;
            std::stack<Node> stack;
            stack.push(mNodes[0]);
            while (!stack.empty()) {
                const Node& node = stack.top();
                stack.pop();
                if ( node.isLeaf() ) {
                    IndexType index = node.index();
                    for (auto i = index.id(); i < index.id() + index.primCount(); ++i) {
                        auto hit = mPrimitives[mPrimIds[i]].intersect(ray, eps);
                        if (hit) {
                            auto [t, u, v] = *hit;
                            if (t < closestT) {
                                closestT = t;
                                closestHit = *hit;
                            }
                        }
                    }
                    index.id();
                    index.primCount();
                } else if ( node.bbox().intersect( ray ) ) {
                    auto nodeIdx = node.index().id();
                    stack.emplace(mNodes[nodeIdx]);
                    stack.emplace(mNodes[nodeIdx + 1]);
                }
            }
            return closestHit;
        }

    protected:
        std::vector<Node> mNodes;
        std::span<Primitive> mPrimitives;
        std::vector<size_t> mPrimIds;
    };
}

#endif //COLLECTION_BVH_HPP