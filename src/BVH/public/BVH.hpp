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
    template <typename Node, typename Primitive>
    class BinnedSAHBuilder;

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
        using HitType = Hit<Type>;
        friend class SweepSAHBuilder<Node, Primitive>;
        friend class BinnedSAHBuilder<Node, Primitive>;

        std::optional<HitType> intersect(const Ray<Type>& ray, Type eps) const {
            std::optional<HitType> closestHit;
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

        std::optional<HitType> intersect16(const Ray<Type>& ray, Type eps) const {
            std::optional<HitType> closestHit;
            Type closestT = std::numeric_limits<Type>::max();

            std::stack<Node> stack;
            stack.push(mNodes[0]);
            while (!stack.empty()) {
                const Node& node = stack.top();
                stack.pop();
                if (node.isLeaf()) {
                    IndexType index = node.index();
                    for (int iBlock = 0; iBlock < index.primCount(); iBlock += 16) {
                        PreTri16 tri16 = mSimdTris[mPrimIds[index.id() + iBlock]];
                        auto hit16 = tri16.intersect(ray);
                        if (hit16) {
                            auto [t, u, v, hitIdx] = *hit16;
                            int primIdx = mPrimIds[index.id() + iBlock + hitIdx];
                            if (t < closestT) {
                                closestT = t;
                                closestHit = {primIdx, t, u, v};
                            }
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

        void fillSimdTris() {
            mSimdTris.resize( mPrimitives.size() );
            for (auto &node: mNodes) {
                if (!node.isLeaf()) continue;
                IndexType index = node.index();
                for (int iBlock = 0; iBlock < index.primCount(); iBlock += 16) {
                    PreTri16 tri16;
                    const int blockSize = std::min(16, static_cast<int>(index.primCount() - iBlock));
                    for (int i = 0; i < blockSize; ++i) {
                        const auto &tri = mPrimitives[mPrimIds[index.id() + iBlock + i]];
                        ((float *) &tri16.p0x)[i] = tri.p0.x;
                        ((float *) &tri16.p0y)[i] = tri.p0.y;
                        ((float *) &tri16.p0z)[i] = tri.p0.z;
                        ((float *) &tri16.e1x)[i] = tri.e1.x;
                        ((float *) &tri16.e1y)[i] = tri.e1.y;
                        ((float *) &tri16.e1z)[i] = tri.e1.z;
                        ((float *) &tri16.e2x)[i] = tri.e2.x;
                        ((float *) &tri16.e2y)[i] = tri.e2.y;
                        ((float *) &tri16.e2z)[i] = tri.e2.z;
                        ((float *) &tri16.Nx)[i] = tri.N.x;
                        ((float *) &tri16.Ny)[i] = tri.N.y;
                        ((float *) &tri16.Nz)[i] = tri.N.z;
                    }

                    for (int i = blockSize; i < 16; ++i) {
                        ((float *) &tri16.p0x)[i] = 0;
                        ((float *) &tri16.p0y)[i] = 0;
                        ((float *) &tri16.p0z)[i] = 0;
                        ((float *) &tri16.e1x)[i] = 0;
                        ((float *) &tri16.e1y)[i] = 0;
                        ((float *) &tri16.e1z)[i] = 0;
                        ((float *) &tri16.e2x)[i] = 0;
                        ((float *) &tri16.e2y)[i] = 0;
                        ((float *) &tri16.e2z)[i] = 0;
                        ((float *) &tri16.Nx)[i] = 0;
                        ((float *) &tri16.Ny)[i] = 0;
                        ((float *) &tri16.Nz)[i] = 0;
                    }
                    mSimdTris[mPrimIds[index.id() + iBlock]] = tri16;
                }
            }
        }

        const Primitive& primitive(size_t idx) const { return mPrimitives[idx]; }

    protected:
        std::vector<Node> mNodes;
        std::span<Primitive> mPrimitives;
        std::vector<PreTri16> mSimdTris;
        std::vector<size_t> mPrimIds;
    };
}

#endif //COLLECTION_BVH_HPP