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

    template <typename Type>
    struct Hit {
        using Vec3 = glm::vec<3, Type>;
        size_t id;
        Type t;
        Type u;
        Type v;
    };

    template <typename NodeType, typename Primitive>
    class BVH {
    public:
        using Node = NodeType;
        using Type = Node::Type;
        using IndexType = Node::IndexType;
        using Hit = Hit<Type>;
        friend class SweepSAHBuilder<Node, Primitive>;

        std::optional<Hit> intersect(const Ray<Type>& ray, Type eps) const {
            std::optional<Hit> closestHit;
            Type closestT = std::numeric_limits<Type>::max();

            std::stack<Node> stack;
            stack.push(mNodes[0]);
            while (!stack.empty()) {
                const Node& node = stack.top();
                stack.pop();
                if (node.isLeaf()) {
                    IndexType index = node.index();
                    for (auto i = index.id(); i < index.id() + index.primCount(); ++i) {
                        auto& primitive = mPrimitives[mPrimIds[i]];
                        auto hit = primitive.intersect(ray, eps);
                        if (!hit) continue;
                        auto [t, u, v] = *hit;
                        if (t < closestT) {
                            closestT = t;
                            closestHit = {mPrimIds[i], t, u, v};
                        }
                    }
                } else if (node.bbox().intersect(ray)) {
                    auto nodeIdx = node.index().id();
                    stack.emplace(mNodes[nodeIdx]);
                    stack.emplace(mNodes[nodeIdx + 1]);
                }
            }
            return closestHit;
        }

        const Primitive& primitive(size_t idx) const { return mPrimitives[idx]; }

    protected:
        std::vector<Node> mNodes;
        std::span<Primitive> mPrimitives;
        std::vector<size_t> mPrimIds;
    };
}

#endif //COLLECTION_BVH_HPP