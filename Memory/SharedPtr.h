//
// Created by auser on 2/3/25.
//

#ifndef COLLECTION_SHAREDPTR_H
#define COLLECTION_SHAREDPTR_H

#include "BasicAllocator.h"

class ControlBlock {
public:
    ControlBlock(): mRefCount( 0 ) {}
    ControlBlock( int refCount ): mRefCount( refCount ) {}
    int mRefCount;
};

template <typename Type, typename Allocator = BasicAllocator<Type>>
class SharedPtr {
public:
    constexpr SharedPtr() noexcept: mPtr(0), mControl(), mAlloc() {}

    SharedPtr( Type* ptr ): mPtr(ptr), mControl( new ControlBlock( 1 ) ) {}

    template <typename... Args>
    static SharedPtr make(Args&&... args) {
        Allocator alloc;
        Type* ptr = alloc.allocate( 1 );
        alloc.construct( ptr, std::forward<Args>( args )... );
        return SharedPtr( ptr );
    }

    SharedPtr( const SharedPtr& other ) {
        mPtr = other.mPtr;
        mControl = other.mControl;
        ++mControl->mRefCount;
    }
    SharedPtr& operator=( const SharedPtr& other ) {
        if ( this == &other ) return *this;
        clear();
        mPtr = other.mPtr;
        mControl = other.mControl;
        ++mControl->mRefCount;
        return *this;
    }

    SharedPtr( SharedPtr&& other ) noexcept : mPtr( other.mPtr ), mControl( other.mControl ) {
        other.mPtr = nullptr;
        other.mControl = nullptr;
    }

    SharedPtr& operator=( SharedPtr&& other )  noexcept {
        if ( this == &other ) return *this;
        mAlloc.destroy( mPtr );
        mAlloc.deallocate( mPtr, 1 );
        mAlloc.destroy( mControl );
        mAlloc.deallocate( mControl, 1 );
        mPtr = other.mPtr;
        mControl = other.mControl;
        other.mPtr = nullptr;
        other.mControl = nullptr;
        return *this;
    }

    ~SharedPtr() {
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
        if ( --mControl->mRefCount != 0 ) return;
        mAlloc.destroy( mPtr );
        mAlloc.deallocate( mPtr, 1 );
        delete mControl;
    }

    Type* mPtr;
    ControlBlock* mControl;
    Allocator mAlloc;
};



#endif //COLLECTION_SHAREDPTR_H
