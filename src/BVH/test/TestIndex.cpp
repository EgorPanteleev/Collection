//
// Created by igor on 3/9/26.
//

#include <gtest/gtest.h>
#include "Index.hpp"

using crv::graphics::Index;

TEST(Index, Default_Constructor) {
    Index<32, 8> index;
    EXPECT_EQ(index.id(), 0);
    EXPECT_EQ(index.primCount(), 0);
}

TEST(Index, Constructor1) {
    Index<32, 8> index(123, 200);
    EXPECT_EQ(index.id(), 123);
    EXPECT_EQ(index.primCount(), 200);
}

TEST(Index, Constructor2) {
    Index<32, 8> index(static_cast<size_t>(1267));
    EXPECT_EQ(index.id(), 1267);
    EXPECT_EQ(index.primCount(), 0);
}

TEST(Index, Set) {
    Index<32, 8> index;
    index.setID(12345);
    index.setPrimCount(250);
    EXPECT_EQ(index.id(), 12345);
    EXPECT_EQ(index.primCount(), 250);
}

TEST(Index, LeafInner) {
    Index<32, 8> index;
    index.setPrimCount(250);
    EXPECT_EQ(index.isLeaf(), true);
    index.setPrimCount(0);
    EXPECT_EQ(index.isInner(), true);
}
