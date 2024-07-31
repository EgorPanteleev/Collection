//
// Created by auser on 7/31/24.
//

#ifndef COLLECTION_ITERATOR_H
#define COLLECTION_ITERATOR_H
#include <hip/hip_runtime.h>

template <typename Type>
class Iterator {
public:
    __host__ __device__ Iterator(Type* ptr) : mPtr(ptr) {}

    __host__ __device__ Type& operator*() { return *mPtr; }
    __host__ __device__ const Type& operator*() const { return *mPtr; }

    __host__ __device__ Iterator& operator++() {
        ++mPtr;
        return *this;
    }
    __host__ __device__ Iterator operator++(int) {
        Iterator temp = *this;
        ++(*this);
        return temp;
    }

    __host__ __device__ bool operator==(const Iterator& other) const { return mPtr == other.mPtr; }
    __host__ __device__ bool operator!=(const Iterator& other) const { return mPtr != other.mPtr; }

private:
    Type* mPtr;
};

#endif //COLLECTION_ITERATOR_H
