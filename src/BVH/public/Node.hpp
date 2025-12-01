//
// Created by igor on 11/1/25.
//

#ifndef NODE_HPP
#define NODE_HPP

#include <limits.h>

#include "Index.hpp"
#include "BBox.hpp"

//        Type id() const { return value >> primCountBits; }
// Type primCount() const { return value & maxPrimCount; }
// bool isLeaf() const { return primCount() != 0; }
// bool isInner() const { return !isLeaf(); }
// void setID(size_t id) { set(id, static_cast<size_t>(primCount())); }
// void setPrimCount(size_t primCount) { set(static_cast<size_t>(id()), primCount); }

namespace crv::graphics {
    template<typename T,
        size_t indexBits = sizeof(T) * CHAR_BIT,
        size_t primCountBits = 4>
    class Node {
    public:
        using Type = T;
        using IndexType = Index<indexBits, primCountBits>;
        using BoxType = BBox<Type>;

        Node() = default;

        bool isLeaf() const { return mIndex.isLeaf(); }
        BoxType getBBox() const { return mBBox; }

        void setBBox(const BoxType& bbox) { mBBox = bbox; }
        void setIndex(const IndexType& index) { mIndex = index; }
        //todo impelement intersection
    private:
        IndexType mIndex;
        BoxType mBBox; //std::array<T, Dim * 2> bounds;
    };
}

#endif //NODE_HPP
