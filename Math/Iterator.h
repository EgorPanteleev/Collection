//
// Created by auser on 7/31/24.
//

#ifndef COLLECTION_ITERATOR_H
#define COLLECTION_ITERATOR_H

template <typename Type>
class Iterator {
public:
    Iterator(Type* ptr) : mPtr(ptr) {}

    Type& operator*() { return *mPtr; }
    const Type& operator*() const { return *mPtr; }

    Iterator& operator++() {
        ++mPtr;
        return *this;
    }
    Iterator operator++(int) {
        Iterator temp = *this;
        ++(*this);
        return temp;
    }

    bool operator==(const Iterator& other) const { return mPtr == other.mPtr; }
    bool operator!=(const Iterator& other) const { return mPtr != other.mPtr; }

private:
    Type* mPtr;
};

#endif //COLLECTION_ITERATOR_H
