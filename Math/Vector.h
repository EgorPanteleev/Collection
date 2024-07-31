//
// Created by auser on 7/30/24.
//

#ifndef COLLECTION_VECTOR_H
#define COLLECTION_VECTOR_H
#include "Iterator.h"

template <typename Type>
class Vector {
public:
    __host__ __device__ Vector() : mData(nullptr), mSize(0), mCapacity(0) {}

    __host__ __device__ ~Vector() {
        delete[] mData;
    }

    __host__ __device__ Vector(const Vector& other) {
        mSize = other.mSize;
        mCapacity = other.mCapacity;
        mData = new Type[mCapacity];
        copy(other.mData, other.mData + mSize, mData);
    }

    __host__ __device__ void push_back(const Type& value) {
        if (mSize >= mCapacity) {
            reserve( mCapacity == 0 ? 1 : 2 * mCapacity );
        }
        mData[mSize++] = value;
    }

    __host__ __device__ void pop_back() {
        mData[--mSize] = Type();
    }

    [[nodiscard]] __host__ __device__ unsigned long size() const {
        return mSize;
    }

    __host__ __device__ void clear() {
        for (unsigned long i = 0; i < mSize; ++i) {
            mData[i].~Type();
        }
        mSize = 0;
    }

    __host__ __device__ Iterator<Type> begin() { return Iterator(mData); }

    __host__ __device__ Iterator<Type> end() { return Iterator(mData + mSize); }

    __host__ __device__ Iterator<Type> begin() const { return Iterator(mData); }

    __host__ __device__ Iterator<Type> end() const { return Iterator(mData + mSize); }

    __host__ __device__ Vector<Type>& operator=(const Vector& other) {
        if (this != &other) {
            delete[] mData;
            mSize = other.mSize;
            mCapacity = other.mCapacity;
            mData = new Type[mCapacity];
            copy(other.mData, other.mData + mSize, mData);
        }
        return *this;
    }

    __host__ __device__ Type& operator[](unsigned long index) {
        return mData[index];
    }

    __host__ __device__ const Type& operator[](unsigned long index) const {
        return mData[index];
    }

private:

    __host__ __device__ void reserve(unsigned long newCapacity) {
        if ( newCapacity <= mCapacity ) return;
        Type* newData = new Type[newCapacity];
        copy(mData, mData + mSize, newData);
        delete[] mData;
        mData = newData;
        mCapacity = newCapacity;
    }


    template <typename InputIt, typename OutputIt>
    __host__ __device__ OutputIt copy(InputIt first, InputIt last, OutputIt dest) {
        while (first != last) {
            *dest++ = *first++;
        }
        return dest;
    }

private:
    Type* mData;
    unsigned long mSize;
    unsigned long mCapacity;
};


#endif //COLLECTION_VECTOR_H
