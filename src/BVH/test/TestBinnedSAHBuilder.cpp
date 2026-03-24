//
// Created by igor on 3/24/26.
//

#include <gtest/gtest.h>
#include <random>

#define private public
#define protected public

#include "BinnedSAHBuilder.hpp"
#include "Node.hpp"
#include "Message.hpp"
#include "Triangle.hpp"

using namespace crv::graphics;
using Tri = PrecomputedTriangle<float>;
using Vec3 = Tri::Vec3;
using NodeType = Node<float, 32, 8>;
using Box = NodeType::Box;

std::vector<Tri> randomTriangles(size_t n) {
    std::vector<Tri> res;
    res.reserve(n);

    std::mt19937 gen(43);
    std::uniform_int_distribution dist(-100000, 100000);

    for (size_t i = 0; i < n; ++i) {
        Vec3 a(dist(gen), dist(gen), dist(gen));
        Vec3 b(dist(gen), dist(gen), dist(gen));
        Vec3 c(dist(gen), dist(gen), dist(gen));

        res.emplace_back(a, b, c);
    }

    return res;
}

TEST(BVH, Build) {
    std::vector<Tri> triangles = randomTriangles(10000);
    BinnedSAHBuilder<NodeType, Tri> builder{std::span(triangles)};
    auto bvh = builder.build();
    auto primIds = builder.mPrimIds;
    for (int nodeId = 0; nodeId < bvh.mNodes.size(); ++nodeId) {
        NodeType& node = bvh.mNodes[nodeId];
        Box nodeBox = node.mBBox;
        Box computedBox;
        if (node.mIndex.isInner()) {
            NodeType& leftChild = bvh.mNodes[node.mIndex.id()];
            NodeType& rightChild = bvh.mNodes[node.mIndex.id() + 1];
            computedBox = leftChild.bbox() + rightChild.bbox();
        } else {
            auto primIdsView = primIds | std::views::drop(node.mIndex.id()) | std::views::take(node.mIndex.primCount());
            std::ranges::for_each(primIdsView, [&builder, &computedBox](size_t index) {
                computedBox += builder.mBBoxes[index];
            } );
        }
        EXPECT_EQ(nodeBox, computedBox);
    }
}