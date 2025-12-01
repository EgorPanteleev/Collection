//
// Created by igor on 11/1/25.
//

#ifndef BINNEDSAHBUILDER_HPP
#define BINNEDSAHBUILDER_HPP

#include <algorithm>
#include <span>
#include <expected>
#include <string>
#include <stack>
#include <ranges>

#include "BVH.hpp"
#include "BBox.hpp"

#define MIN_LEAF_SIZE 8

namespace crv::graphics {
    template <typename Node, size_t binCount = 8>
    class BinnedSAHBuilder {
    public:
        using Type = typename Node::Type;
        using BoxType = BBox<typename Node::Type>;
        using Vec3 = typename BoxType::Vec3;
        using IndexType = typename Node::IndexType;
        struct Item {
            size_t nodeId;
            size_t begin;
            size_t end;
            size_t size() const { return end - begin; }
        };

        BinnedSAHBuilder(std::span<BoxType> bboxes, std::span<Vec3> centers): mBBoxes(bboxes), mCenters(centers) {}

        struct Split {
            size_t binID;
            Type cost;
            size_t axis;
        };

        struct Bin {
            Bin() = default;

            Type getCost() const {  } //TODO
            void add(const BoxType& other, size_t cnt = 1) { bbox += other; primCount += cnt; }
            void add(const Bin& bin) { add(bin.bbox, bin.primCount); }
            BoxType bbox;
            size_t primCount;
        };


        std::expected<size_t, std::string> trySplit() {

        }

        BVH<Node> build() {
            const auto size = mBBoxes.size();
            BVH<Node> bvh;
            bvh.mNodes.reserve(2 * size);
            bvh.mNodes.emplace_back();
            bvh.mNodes.back().setBBox(computeBBox(0, size));
            std::stack<Item> stack;
            stack.push({0, 0, size});
            while (!stack.empty()) {
                Item item = stack.top();
                stack.pop();
                Node& node = bvh.mNodes[item.nodeId];
                if (item.size() > MIN_LEAF_SIZE) {
                    if (auto status = trySplit(); status.has_value()) {
                        size_t splitPos = status.value();
                        size_t firstChild = bvh.mNodes.size();
                        node.setIndex(IndexType(firstChild));
                        bvh.mNodes.resize(firstChild + 2);

                        BoxType firstBBox  = computeBBox(item.begin, splitPos);
                        BoxType secondBBox = computeBBox(splitPos, item.end);
                        auto firstRange  = std::make_pair(item.begin, splitPos);
                        auto secondRange = std::make_pair(splitPos, item.end);

                        if (firstBBox.getHalfArea() < secondBBox.getHalfArea()) {
                            std::swap(firstBBox, secondBBox);
                            std::swap(firstRange, secondRange);
                        }
                        Item firstItem (firstChild + 0, firstRange. first, firstRange. second);
                        Item secondItem(firstChild + 1, secondRange.first, secondRange.second);
                        if (firstItem.size() < secondItem.size()) std::swap(firstItem, secondItem);
                        stack.push(firstItem);
                        stack.push(secondItem);
                        continue;
                    }
                }
                node.setIndex(IndexType(item.begin, item.size()));
            }

            bvh.mPrimIds = mPrimIds;
            bvh.mNodes.shrink_to_fit();
            return bvh;
        }
    private:
        BoxType computeBBox(size_t begin, size_t end) {
            BoxType res{};
            std::ranges::for_each(std::ranges::subrange(mBBoxes.begin() + begin, mBBoxes.begin() + end),
                [&res](const BoxType& box) {
                res += box;
            });
            return res;
        }

        std::span<BoxType> mBBoxes;
        std::span<Vec3> mCenters;
        std::vector<size_t> mPrimIds;
    };
}

#endif //BINNEDSAHBUILDER_HPP
