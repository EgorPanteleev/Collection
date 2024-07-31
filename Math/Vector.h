//
// Created by auser on 7/30/24.
//

#ifndef COLLECTION_VECTOR_H
#define COLLECTION_VECTOR_H
#include "Iterator.h"

template <typename Type>
class Vector {
public:
    Vector() : mData(nullptr), mSize(0), mCapacity(0) {}
    
    ~Vector() {
        delete[] mData;
    }

    Vector(const Vector& other) {
        mSize = other.mSize;
        mCapacity = other.mCapacity;
        mData = new Type[mCapacity];
        copy(other.mData, other.mData + mSize, mData);
    }

    void push_back(const Type& value) {
        if (mSize >= mCapacity) {
            reserve( mCapacity == 0 ? 1 : 2 * mCapacity );
        }
        mData[mSize++] = value;
    }

    void pop_back() {
        mData[--mSize] = Type();
    }

    [[nodiscard]] unsigned long size() const {
        return mSize;
    }

    void clear() {
        for (unsigned long i = 0; i < mSize; ++i) {
            mData[i].~Type();
        }
        mSize = 0;
    }

    Iterator<Type> begin() { return Iterator(mData); }

    Iterator<Type> end() { return Iterator(mData + mSize); }

    Iterator<Type> begin() const { return Iterator(mData); }

    Iterator<Type> end() const { return Iterator(mData + mSize); }

    Vector<Type>& operator=(const Vector& other) {
        if (this != &other) {
            delete[] mData;
            mSize = other.mSize;
            mCapacity = other.mCapacity;
            mData = new Type[mCapacity];
            copy(other.mData, other.mData + mSize, mData);
        }
        return *this;
    }

    Type& operator[](unsigned long index) {
        return mData[index];
    }

    const Type& operator[](unsigned long index) const {
        return mData[index];
    }

private:

    void reserve(unsigned long newCapacity) {
        if ( newCapacity <= mCapacity ) return;
        Type* newData = new Type[newCapacity];
        copy(mData, mData + mSize, newData);
        delete[] mData;
        mData = newData;
        mCapacity = newCapacity;
    }


    template <typename InputIt, typename OutputIt>
    OutputIt copy(InputIt first, InputIt last, OutputIt dest) {
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
