//
// Created by igor on 3/9/26.
//

#ifndef COLLECTION_NODE_HPP
#define COLLECTION_NODE_HPP

#include "Index.hpp"

namespace crv::graphics {
    template <typename T,
              size_t indexBits = sizeof(T) * CHAR_BIT,
              size_t primBits = 4>
    class Node {
    public:
        using Type = T;
        using IndexType = Index<indexBits, primBits>;
        using Box = BBox<Type>;
        explicit Node(const Box& bbox, const IndexType& index): mBBox(bbox), mIndex(index) {}
        explicit Node(const Box& bbox): mBBox(bbox), mIndex() {}

        void setBBox(const Box& bbox) { mBBox = bbox; }
        void setIndex(const IndexType& index) { mIndex = index; }

        Box bbox() const { return mBBox; }
        bool isLeaf() const { return mIndex.isLeaf(); }
    protected:
        Box mBBox;
        IndexType mIndex;
    };
}

#endif //COLLECTION_NODE_HPP