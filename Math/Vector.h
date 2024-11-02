//
// Created by auser on 10/31/24.
//

#ifndef VECTOR_VECTOR_H
#define VECTOR_VECTOR_H
#include "BasicAllocator.h"

template <typename Type, typename Allocator = BasicAllocator<Type>>
class Vector {
public:

    using Reference = Type&;
    using ConstReference = const Type&;
    using ValueType = Type;
    using AllocatorType = Allocator;

    Vector(): mAlloc(), mData(nullptr), mSize(0), mCap(0)  {}


    Vector( size_t n ): mAlloc(), mData(nullptr), mSize(0), mCap(0)  {
        reserve( n );
    }

    Vector( size_t n, const Type& value ): mAlloc(), mData(nullptr), mSize(0), mCap(n)  {
        resize( n, value );
    }

    Vector( size_t n, Type&& value ): mAlloc(), mData(nullptr), mSize(0), mCap(0)  {
        resize( n, std::move( value ) );
    }

    ~Vector() {
        for ( int i = 0; i < mSize; ++i ) {
            mAlloc.destroy( mData + i );
        }
        mAlloc.deallocate( mData, mCap );
    }
    void reserve( size_t newCap ) {
        if ( newCap < mCap ) return;
        Type* newData = mAlloc.allocate( newCap );
        for ( int i = 0; i < mSize; ++i ) {
            mAlloc.construct( &newData[i], std::move( mData[i] ) );
        }
        for ( int i = 0; i < mSize; ++i ) {
            mAlloc.destroy( mData + i );
        }
        mAlloc.deallocate( mData, mCap );
        mData = newData;
        mCap = newCap;
    }

    void resize( size_t newSize, const Type& value = Type() ) {
        if ( newSize <= mSize ) {
            mSize = newSize;
            return;
        }

        if ( newSize > mCap ) {
            reserve( newSize );
        }

        for ( int i = mSize; i < newSize; ++i ) {
            mAlloc.construct( &mData[i], value );
        }
        mSize = newSize;
    }

    void push_back( const Type& value ) {
        if ( mSize == mCap ) {
            reserve( mCap == 0 ? 1 : mCap * 2 );
        }

        mAlloc.construct( &mData[ mSize++ ], value );
    }

    void push_back( Type&& value ) {
        if ( mSize == mCap ) {
            reserve( mCap == 0 ? 1 : mCap * 2 );
        }

        mAlloc.construct( &mData[ mSize++ ], std::move( value ) );
    }

    void pop_back() {
        --mSize;
    }

    template<typename... Args>
    void emplace_back( Args&&... args ) {
        if ( mSize == mCap ) {
            reserve( mCap == 0 ? 1 : mCap * 2 );
        }
        mAlloc.construct( &mData[ mSize++ ], std::forward<Args>( args )... );
    }

    Reference operator[]( size_t ind ) {
        return mData[ind];
    }

    ConstReference operator[]( size_t ind ) const {
        return mData[ind];
    }

    Reference at( size_t ind ) {
        if ( ind >= mSize ) throw std::out_of_range("Index out of range!");
        return mData[ind];
    }

    ConstReference at( size_t ind ) const {
        if ( ind >= mSize ) throw std::out_of_range("Index out of range!");
        return mData[ind];
    }

    Reference front() {
        return mData[0];
    }

    ConstReference front() const {
        return mData[0];
    }

    Reference back() {
        return mData[mSize - 1];
    }

    ConstReference back() const {
        return mData[mSize - 1];
    }

    Type* data() {
        return mData;
    }

    const Type* data() const {
        return mData;
    }

    bool empty() const {
        return mSize == 0;
    }

    size_t size() const {
        return mSize;
    }

    size_t capacity() const {
        return mCap;
    }

    void clear() {
        mSize = 0;
    }

    class Iterator {
    public:
        Iterator( Type* ptr ): ptr( ptr ) {}

        Iterator& operator++() {
            ++ptr;
            return *this;
        }

        Iterator& operator+=( size_t n ) {
            ptr += n;
            return *this;
        }

        Iterator& operator--() {
            --ptr;
            return *this;
        }

        Iterator& operator-=( size_t n ) {
            ptr -= n;
            return *this;
        }

        Type& operator*() {
            return *ptr;
        }

        const Type& operator*() const {
            return *ptr;
        }

        Type* operator->() {
            return ptr;
        }

        const Type* operator->() const {
            return ptr;
        }

        int operator-( const Iterator& other ) const {
            return ptr - other.ptr;
        }

        bool operator==( const Iterator& other ) const {
            return ptr == other.ptr;
        }

        bool operator!=( const Iterator& other ) const {
            return ptr != other.ptr;
        }

        bool operator>=( const Iterator& other ) const {
            return ptr >= other.ptr;
        }

        bool operator<=( const Iterator& other ) const {
            return ptr < other.ptr;
        }

        bool operator>( const Iterator& other ) const {
            return ptr > other.ptr;
        }

        bool operator<( const Iterator& other ) const {
            return ptr < other.ptr;
        }

    private:
        Type* ptr;
    };

    Iterator begin() {
        return Iterator( mData );
    }

    const Iterator begin() const {
        return Iterator( mData );
    }

    Iterator end() {
        return Iterator( mData + mSize );
    }

    const Iterator end() const {
        return Iterator( mData + mSize );
    }

private:
    Allocator mAlloc;
    Type* mData;
    size_t mSize;
    size_t mCap;
};

#endif //VECTOR_VECTOR_H
