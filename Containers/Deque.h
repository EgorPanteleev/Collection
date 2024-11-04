//
// Created by auser on 11/3/24.
//

#ifndef COLLECTION_DEQUE_H
#define COLLECTION_DEQUE_H
#include "BasicAllocator.h"
#include "Vector.h"

template <typename Type, typename Allocator = BasicAllocator<Type>>
class Deque {
public:

private:
    void expand_map() {

    }

    static const size_t blockSize = 8;
    Vector<Type*> mData;
    size_t mStartBlock, mEndBlock;
    size_t mStartIndex, mEndIndex;
    size_t mSize;
};

#endif //COLLECTION_DEQUE_H
