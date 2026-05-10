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
    class BVH16;

    template <typename NodeType, typename Primitive>
    class BVH {
    public:
        using Node = NodeType;
        using Type = Node::Type;
        using IndexType = Node::IndexType;
        using HitType = Hit<Type>;
        friend class SweepSAHBuilder<Node, Primitive>;
        friend class BinnedSAHBuilder<Node, Primitive>;
        friend class BVH16<Node, Primitive>;

        BVH() = default;
        BVH& operator=(const BVH& other) = default;
        BVH(const BVH& other) = default;

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

        const Primitive& primitive(size_t idx) const { return mPrimitives[idx]; }

        std::vector<Node>& nodes() { return mNodes; }
        std::vector<uint32_t>& primIds() { return mPrimIds; }

    protected:
        std::vector<Node> mNodes;
        std::span<Primitive> mPrimitives;
        std::vector<uint32_t> mPrimIds;
    };

    template <typename NodeType, typename Primitive>
    class BVH16: public BVH<NodeType, Primitive> {
    public:
        using BVH<NodeType, Primitive>::mPrimitives;
        using BVH<NodeType, Primitive>::mPrimIds;
        using BVH<NodeType, Primitive>::mNodes;
        using IndexType = BVH<NodeType, Primitive>::IndexType;
        using HitType = BVH<NodeType, Primitive>::HitType;
        using Type = BVH<NodeType, Primitive>::Type;
        using Node = BVH<NodeType, Primitive>::Node;

        BVH16() = default;
        explicit BVH16(const BVH<NodeType, Primitive>& bvh) {
            mNodes = bvh.mNodes;
            mPrimitives = bvh.mPrimitives;
            mPrimIds = bvh.mPrimIds;
            fillSimdTris();
        }
        BVH16& operator=(const BVH<NodeType, Primitive>& bvh) {
            mNodes = bvh.mNodes;
            mPrimitives = bvh.mPrimitives;
            mPrimIds = bvh.mPrimIds;
            fillSimdTris();
            return *this;
        }
        void fillSimdTris() {
            mSimdTris.resize(mPrimitives.size());
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

        std::optional<HitType> intersect16(const Ray<Type> &ray, Type eps) const {
            std::optional<HitType> closestHit;
            Type closestT = std::numeric_limits<Type>::max();
            std::stack<uint32_t> stack;
            stack.push(0);
            while (!stack.empty()) {
                uint32_t nodeIdx = stack.top();
                stack.pop();
                const Node &node = mNodes[nodeIdx];
                auto bboxHit = node.bbox().intersect(ray);
                if (!bboxHit || *bboxHit >= closestT) continue;
                if (node.isLeaf()) {
                    auto index = node.index();
                    uint32_t base = index.id();
                    for (uint32_t i = 0; i < index.primCount(); i += 16) {
                        PreTri16 tri16 = mSimdTris[mPrimIds[base + i]];
                        auto hit16 = tri16.intersect(ray);
                        if (hit16) {
                            auto [t, u, v, lane] = *hit16;
                            uint32_t primIdx = mPrimIds[base + i + lane];
                            if (t < closestT) {
                                closestT = t;
                                closestHit = HitType{primIdx, t, u, v};
                            }
                        }
                    }
                } else {
                    uint32_t leftIdx = node.index().id();
                    uint32_t rightIdx = leftIdx + 1;
                    const Node &left = mNodes[leftIdx];
                    const Node &right = mNodes[rightIdx];
                    auto tLeft = left.bbox().intersect(ray);
                    auto tRight = right.bbox().intersect(ray);
                    bool hitLeft = tLeft && *tLeft < closestT;
                    bool hitRight = tRight && *tRight < closestT;
                    if (hitLeft && hitRight) {
                        if (*tLeft < *tRight) {
                            stack.push(rightIdx);
                            stack.push(leftIdx);
                        } else {
                            stack.push(leftIdx);
                            stack.push(rightIdx);
                        }
                    } else if (hitLeft) {
                        stack.push(leftIdx);
                    } else if (hitRight) {
                        stack.push(rightIdx);
                    }
                }
            }
            return closestHit;
        }

    protected:
        std::vector<PreTri16> mSimdTris;
    };
}

#endif //COLLECTION_BVH_HPP