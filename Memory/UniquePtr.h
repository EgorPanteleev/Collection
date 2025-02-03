//
// Created by auser on 2/3/25.
//

#ifndef COLLECTION_UNIQUEPTR_H
#define COLLECTION_UNIQUEPTR_H


#include <iostream>
#include <memory>
#include "BasicAllocator.h"

template <typename Type, typename Allocator = BasicAllocator<Type>>
class UniquePtr {
public:
    constexpr UniquePtr() noexcept: mPtr(0), mAlloc() {}

    UniquePtr( Type* ptr ): mPtr(ptr) {}

    template <typename... Args>
    static UniquePtr make(Args&&... args) {
        Allocator alloc;
        Type* ptr = alloc.allocate( 1 );
        alloc.construct( ptr, std::forward<Args>( args )... );
        return UniquePtr( ptr );
    }

    UniquePtr( const UniquePtr& ) = delete;

    UniquePtr& operator=( const UniquePtr& ) = delete;

    UniquePtr( UniquePtr&& other ) noexcept: mPtr( other.mPtr ) {
        other.mPtr = nullptr;
    }

    UniquePtr& operator=( UniquePtr&& other ) noexcept {
        if ( this == &other ) return *this;
        clear();
        mPtr = other.mPtr;
        other.mPtr = nullptr;
        return *this;
    }

    ~UniquePtr() {
        clear();
    }

    Type& operator*() {
        return *mPtr;
    }

    const Type& operator*() const {
        return *mPtr;
    }

    Type* operator->() {
        return mPtr;
    }

    const Type* operator->() const {
        return mPtr;
    }
private:
    void clear() {
        mAlloc.destroy( mPtr );
        mAlloc.deallocate( mPtr, 1 );
    }

    Type* mPtr;
    Allocator mAlloc;
};


#endif //COLLECTION_UNIQUEPTR_H
