//
// Created by igor on 3/24/26.
//

#ifndef COLLECTION_BINNEDSAHBUILDER_HPP
#define COLLECTION_BINNEDSAHBUILDER_HPP

#include "BVH.hpp"
#include "BBox.hpp"
#include "Message.hpp"

#include <algorithm>
#include <span>
#include <stack>

#ifndef BIN_COUNT
#define BIN_COUNT 16
#endif

namespace crv::graphics {
    template <typename Node, typename Primitive>
    class BinnedSAHBuilder {
    public:
        using BvhType = BVH<Node, Primitive>;
        using Type = Node::Type;
        using Box = BBox<Type>;
        using Vec3 = Box::Vec3;
        using IndexType = Node::IndexType;
        BinnedSAHBuilder(std::span<Primitive> primitives): mPrimitives(primitives) {
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
            size_t id;
            Type cost;
            Box leftBox;
            Box rightBox;
        };

        struct Bin {
            void add(const Box& box, const size_t numPrims = 1) {
                bbox += box;
                primCount += numPrims;
            }
            void add(const Bin& bin) { add(bin.bbox, bin.primCount); }
            Bin& operator+=(const Bin& other) { add(other); return *this; }

            Box bbox;
            size_t primCount = 0;
        };

        BvhType build() {
            if (mBBoxes.empty()) return {};
            const size_t size = mBBoxes.size();
            mPrimIds = std::views::iota(static_cast<uint32_t>(0), size) | std::ranges::to<std::vector<uint32_t>>();
            BvhType bvh;
            bvh.mPrimitives = mPrimitives;
            std::stack<NodeData> stack;
            bvh.mNodes.reserve(size * 2);
            bvh.mNodes.emplace_back(computeBBox(0, size));
            stack.emplace(0, 0, size);

            while (!stack.empty()) {
                NodeData data = stack.top();
                size_t primSize = data.size();
                stack.pop();
                Node& node = bvh.mNodes[data.nodeId];
                SplitData bestSplit = split(data, node);
                Type leafCost = primSize * INTERSECTION_COST;
                if (primSize <= IndexType::maxPrim() and bestSplit.cost >= leafCost) {
                    node.setIndex(IndexType{data.begin, primSize});
                    continue;
                }
                Type min = node.bbox().min[bestSplit.axis];
                Type max = node.bbox().max[bestSplit.axis];
                Type binStep = (max - min) / BIN_COUNT;
                if (binStep <= 0) continue;

                auto beginIt = mPrimIds.begin() + data.begin;
                auto endIt   = mPrimIds.begin() + data.end;
                auto midIt = std::partition(beginIt, endIt,
                    [&](size_t primId){
                        const Type center = mCenters[primId][bestSplit.axis];
                        size_t binId = (center - min) / binStep;
                        binId = std::clamp(binId, static_cast<size_t>(0), static_cast<size_t>(BIN_COUNT) - 1);
                        return binId <= bestSplit.id;
                    });
                size_t splitId = std::distance(mPrimIds.begin(), midIt);
                if (splitId == data.begin || splitId == data.end) {
                    if (primSize <= IndexType::maxPrim()) {
                        node.setIndex(IndexType{data.begin, primSize});
                        continue;
                    }
                    splitId = data.begin + primSize / 2;
                    bestSplit.leftBox  = computeBBox(data.begin, splitId);
                    bestSplit.rightBox = computeBBox(splitId, data.end);
                }

                std::pair<size_t, size_t> leftRange = {data.begin, splitId};
                std::pair<size_t, size_t> rightRange = {splitId, data.end};
                if (bestSplit.leftBox.getHalfArea() > bestSplit.rightBox.getHalfArea()) {
                    std::swap(bestSplit.leftBox, bestSplit.rightBox);
                    std::swap(leftRange, rightRange);
                }
                size_t childIdx = bvh.mNodes.size();
                node.setIndex(IndexType{childIdx});

                stack.emplace(childIdx, leftRange.first, leftRange.second);
                bvh.mNodes.emplace_back(bestSplit.leftBox);

                stack.emplace(childIdx + 1, rightRange.first, rightRange.second);
                bvh.mNodes.emplace_back(bestSplit.rightBox);
            }
            bvh.mPrimIds = mPrimIds;
            return bvh;
        }

        SplitData split(const NodeData& nodeData, const Node& node) {
            SplitData bestSplit;
            bestSplit.cost = std::numeric_limits<Type>::max();
            for (int axis = 0; axis < 3; ++axis) {
                std::array<Bin, BIN_COUNT> bins;
                Vec3 min = node.bbox().min;
                Vec3 max = node.bbox().max;
                Type binStep = (max[axis] - min[axis]) / BIN_COUNT;
                for (size_t i = nodeData.begin; i < nodeData.end; ++i) {
                    const Type &center = mCenters[mPrimIds[i]][axis];
                    const Box &bbox = mBBoxes[mPrimIds[i]];
                    size_t binId = (center - min[axis]) / binStep;
                    binId = std::clamp(binId, static_cast<size_t>(0), static_cast<size_t>(BIN_COUNT) - 1);
                    bins[binId].add(bbox);
                }
                std::array<Bin, BIN_COUNT> prefix;
                prefix[0] = bins[0];
                for (size_t i = 1; i < BIN_COUNT; ++i) {
                    prefix[i] = prefix[i - 1];
                    prefix[i].add(bins[i]);
                }

                std::array<Bin, BIN_COUNT> suffix;
                suffix[BIN_COUNT - 1] = bins[BIN_COUNT - 1];
                for (int i = BIN_COUNT - 2; i >= 0; --i) {
                    suffix[i] = suffix[i + 1];
                    suffix[i].add(bins[i]);
                }
                Type parentArea = node.bbox().getHalfArea();

                for (size_t binId = 0; binId < BIN_COUNT - 1; ++binId) {
                    const Bin &leftBin = prefix[binId];
                    const Bin &rightBin = suffix[binId + 1];
                    const size_t leftCount = leftBin.primCount;
                    const size_t rightCount = rightBin.primCount;
                    if (!leftCount or !rightCount) continue;
                    const Type leftArea = leftBin.bbox.getHalfArea();
                    const Type rightArea = rightBin.bbox.getHalfArea();
                    Type cost = TRAVERSAL_COST +
                                leftArea / parentArea * (leftCount * INTERSECTION_COST) +
                                rightArea / parentArea * (rightCount * INTERSECTION_COST);
                    if (cost >= bestSplit.cost) continue;
                    bestSplit = {
                        .axis = static_cast<uint32_t>(axis),
                        .id = binId,
                        .cost = cost,
                        .leftBox = leftBin.bbox,
                        .rightBox = rightBin.bbox
                    };
                }
            }
            return bestSplit;
        }
    protected:
        Box computeBBox(const size_t begin, const size_t end) const {
            Box res;
            for ( size_t i = begin; i < end; ++i ) {
                res += mBBoxes[mPrimIds[i]];
            }
            return res;
        }

        static constexpr int TRAVERSAL_COST = 1;
        static constexpr int INTERSECTION_COST = 1;

        std::span<Primitive> mPrimitives;
        std::vector<Box> mBBoxes;
        std::vector<Vec3> mCenters;
        std::vector<uint32_t> mPrimIds;
    };
}

#endif //COLLECTION_BINNEDSAHBUILDER_HPP