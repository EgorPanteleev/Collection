//
// Created by igor on 3/10/26.
//

#ifndef COLLECTION_SWEEPSAHBUILDER_H
#define COLLECTION_SWEEPSAHBUILDER_H


#include "BVH.hpp"
#include "BBox.hpp"
#include "Message.hpp"

#include <algorithm>
#include <span>
#include <stack>

namespace crv::graphics {
    template <typename Node, typename Primitive>
    class SweepSAHBuilder {
    public:
        using BvhType = BVH<Node, Primitive>;
        using Type = Node::Type;
        using Box = BBox<Type>;
        using Vec3 = Box::Vec3;
        using IndexType = Node::IndexType;
        SweepSAHBuilder(std::span<Primitive> primitives): mPrimitives(primitives) {
            mBBoxes.reserve(primitives.size());
            mCenters.reserve(primitives.size());
            std::ranges::for_each(primitives, [this](const Primitive& prim) {
                mBBoxes.push_back(prim.bbox());
                mCenters.push_back(prim.center());
            });
        }

        struct NodeData {
            size_t nodeId;
            size_t begin;
            size_t end;
            size_t size() const { return end - begin; }
        };

        struct SplitData {
            size_t axis;
            size_t idx;
            Type cost;
        };

        BvhType build() {
            if (mBBoxes.empty()) return {};
            size_t size = mBBoxes.size();
            for (int axis = 0; axis < 3; ++axis) {
                mIndexesPerAxis[axis] = std::views::iota(static_cast<uint32_t>(0), size) | std::ranges::to<std::vector<uint32_t>>();
                std::ranges::sort(mIndexesPerAxis[axis], [this, axis](size_t idx1, size_t idx2) {
                    return mCenters[idx1][axis] < mCenters[idx2][axis];
                });
            }

            BvhType bvh;
            bvh.mPrimitives = mPrimitives;
            std::stack<NodeData> stack;
            bvh.mNodes.reserve(size * 2);
            bvh.mNodes.emplace_back(computeBBox(0, size));
            stack.emplace(0, 0, size);
            while (!stack.empty()) {
                NodeData data = stack.top();
                size_t nodeSize = data.size();
                stack.pop();
                Node& node = bvh.mNodes[data.nodeId];
                if (nodeSize < IndexType::maxPrim()) {
                    node.setIndex(IndexType{data.begin, nodeSize});
                    continue;
                }
                //split
                SplitData bestSplit;
                bestSplit.cost = std::numeric_limits<Type>::max();
                for (size_t axis = 0; axis < 3; ++axis) {
                    std::vector<Box> prefix(nodeSize);
                    std::vector<Box> suffix(nodeSize);

                    prefix[0] = mBBoxes[mIndexesPerAxis[axis][data.begin]];
                    for (size_t i = 1; i < nodeSize; ++i) {
                        prefix[i] = prefix[i - 1] + mBBoxes[mIndexesPerAxis[axis][data.begin + i]];
                    }

                    suffix[nodeSize - 1] = mBBoxes[mIndexesPerAxis[axis][data.end - 1]];
                    for (int i = static_cast<int>(nodeSize) - 2; i >= 0; --i) {
                        suffix[i] = suffix[i + 1] + mBBoxes[mIndexesPerAxis[axis][data.begin + i]];
                    }
                    Type parentArea = node.bbox().getHalfArea();

                    for (size_t idx = data.begin + 1; idx < data.end; ++idx) {
                        size_t leftCount = idx - data.begin;
                        size_t rightCount = data.end - idx;
                        Type leftArea = prefix[leftCount - 1].getHalfArea();
                        Type rightArea = suffix[leftCount].getHalfArea();

                        Type cost = TRAVERSAL_COST +
                                    leftArea  / parentArea * (leftCount  * INTERSECTION_COST) +
                                    rightArea / parentArea * (rightCount * INTERSECTION_COST);
                        if (cost >= bestSplit.cost) continue;
                        bestSplit = {
                            .axis = axis,
                            .idx = idx,
                            .cost = cost
                        };
                    }
                }
                if (bestSplit.cost > nodeSize * INTERSECTION_COST and nodeSize < IndexType::maxPrim() ) { //not sure
                    node.setIndex(IndexType{data.begin, nodeSize});
                    continue;
                }
                std::vector isLeft(size, false);
                for (int i = 0; i < bestSplit.idx; ++i) {
                    size_t index = mIndexesPerAxis[bestSplit.axis][i];
                    isLeft[index] = true;
                }
                for (int axis = 0; axis < 3; ++axis) {
                    if (axis == bestSplit.axis) continue;
                    auto left = mIndexesPerAxis[axis] |
                        std::views::filter([&isLeft](size_t index) -> bool { return isLeft[index]; });
                    auto right = mIndexesPerAxis[axis] |
                        std::views::filter([&isLeft](size_t index) -> bool { return !isLeft[index]; });

                    std::vector<uint32_t> tmp;
                    tmp.insert(tmp.end(), left.begin(), left.end());
                    tmp.insert(tmp.end(), right.begin(), right.end());
                    mIndexesPerAxis[axis] = std::move(tmp);
                }

                //if (firstBBox.getHalfArea() < secondBBox.getHalfArea()) {
                //std::swap(firstBBox, secondBBox);
                //std::swap(firstRange, secondRange);
                //if (firstItem.size() < secondItem.size()) std::swap(firstItem, secondItem);
                size_t childIdx = bvh.mNodes.size();
                node.setIndex(IndexType{childIdx});
                stack.emplace(childIdx, data.begin, bestSplit.idx);
                bvh.mNodes.emplace_back(computeBBox(data.begin, bestSplit.idx, bestSplit.axis));
                stack.emplace(childIdx + 1, bestSplit.idx, data.end);
                bvh.mNodes.emplace_back(computeBBox(bestSplit.idx, data.end, bestSplit.axis));
            }
            bvh.mPrimIds = mIndexesPerAxis[0];
            return bvh;
        }
    protected:
        Box computeBBox(size_t begin, size_t end, size_t axis = 0) const {
            Box res;
            for ( size_t i = begin; i < end; ++i ) {
                res += mBBoxes[mIndexesPerAxis[axis][i]];
            }
            return res;
        }

        static constexpr int TRAVERSAL_COST = 1;
        static constexpr int INTERSECTION_COST = 1;

        std::span<Primitive> mPrimitives;
        std::vector<Box> mBBoxes;
        std::vector<Vec3> mCenters;
        std::array<std::vector<uint32_t>, 3> mIndexesPerAxis;
    };
}

#endif //COLLECTION_SWEEPSAHBUILDER_H