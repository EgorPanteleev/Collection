//
// Created by auser on 10/31/24.
//

#ifndef VECTOR_VECTOR_H
#define VECTOR_VECTOR_H
#include "BasicAllocator.h"

#include "SystemUtils.h"
#include "Types.h"

#if HIP_ENABLED
#include "hip/hip_runtime.h"
#endif

template <typename Type, typename Allocator = BasicAllocator<Type>>
class Vector {
public:

    using Reference = Type&;
    using ConstReference = const Type&;
    using ValueType = Type;
    using AllocatorType = Allocator;

    HOST_DEVICE Vector(): mAlloc(), mData(nullptr), mSize(0), mCap(0)  {}

    HOST_DEVICE Vector( size_t n, const Type& value = Type() ): mAlloc(), mData(nullptr), mSize(0), mCap(n)  {
        resize( n, value );
    }

    HOST_DEVICE Vector( size_t n, Type&& value ): mAlloc(), mData(nullptr), mSize(0), mCap(0)  {
        resize( n, std::move( value ) );
    }

    HOST_DEVICE Vector( const Vector& other ): mSize(other.mSize), mCap(other.mCap), mAlloc(other.mAlloc)  {
        mData = mAlloc.allocate( mCap );
        for ( size_t i = 0; i < mSize; ++i ) {
            mAlloc.construct( mData + i, other.mData[i] );
        }
    }

    Vector& operator=( const Vector& other ) {
        if ( this == &other ) return *this;
        if ( mCap < other.mSize ) {
            for ( size_t i = 0; i < mSize; ++i ) {
                if ( mData ) mAlloc.destroy( mData + i );
            }
            mAlloc.deallocate( mData, mCap );
            mData = mAlloc.allocate( other.mCap );
            mCap = other.mCap;
        }
        mSize = other.mSize;
        for ( size_t i = 0; i < mSize; ++i ) {
            mAlloc.construct( mData + i, other.mData[i] );
        }
        return *this;
    }

    Vector( Vector&& other ): mData(other.mData), mSize(other.mSize), mCap(other.mCap), mAlloc(std::move(other.mAlloc)) {
        other.mData = nullptr;
        other.clear();
    }

    Vector& operator=( Vector&& other ) {
        if ( this == &other ) return *this;

        for ( size_t i = 0; i < mSize; ++i ) {
            if ( mData ) mAlloc.destroy( mData + i );
        }
        mAlloc.deallocate( mData, mCap );

        mCap = other.mCap;
        mSize = other.mSize;

        mData = other.mData;

        mAlloc = std::move( other.mAlloc );

        other.mData = nullptr;
        other.clear();

        return *this;
    }

    HOST_DEVICE ~Vector() {
        for ( size_t i = 0; i < mSize; ++i ) {
            if ( mData ) mAlloc.destroy( mData + i );
        }
        mAlloc.deallocate( mData, mCap );
    }
    HOST_DEVICE void reserve( size_t newCap ) {
        if ( newCap <= mCap ) return;
        Type* newData = mAlloc.allocate( newCap );
        for ( size_t i = 0; i < mSize; ++i ) {
            mAlloc.construct( newData + i, std::move( mData[i] ) );
        }
        for ( size_t i = 0; i < mSize; ++i ) {
            if ( mData ) mAlloc.destroy( mData + i );
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

        for ( size_t i = mSize - 1; i < newSize; ++i ) {
            mAlloc.construct( mData + i, value );
        }
        mSize = newSize;
    }

    HOST_DEVICE void push_back( const Type& value ) {
        if ( mSize == mCap ) {
            reserve( mCap == 0 ? 1 : mCap * 2 );
        }

        mAlloc.construct( mData + mSize++, value );
    }

    HOST_DEVICE void push_back( Type&& value ) {
        if ( mSize == mCap ) {
            reserve( mCap == 0 ? 1 : mCap * 2 );
        }

        mAlloc.construct( mData + mSize++, std::move( value ) );
    }

    HOST_DEVICE void pop_back() {
        --mSize;
    }

    template<typename... Args>
    void emplace_back( Args&&... args ) {
        if ( mSize == mCap ) {
            reserve( mCap == 0 ? 1 : mCap * 2 );
        }
        mAlloc.construct( mData + mSize++, std::forward<Args>( args )... );
    }

    HOST_DEVICE Reference operator[]( size_t ind ) {
        return mData[ind];
    }

    HOST_DEVICE ConstReference operator[]( size_t ind ) const {
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

    HOST_DEVICE Reference back() {
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

    HOST_DEVICE bool empty() const {
        return mSize == 0;
    }

    HOST_DEVICE size_t size() const {
        return mSize;
    }

    HOST_DEVICE size_t capacity() const {
        return mCap;
    }

    void clear() {
        mSize = 0;
    }

    void swap( Vector& other ) {
        std::swap( mSize, other.mSize );
        std::swap( mCap, other.mCap );
        std::swap( mAlloc, other.mAlloc );
        std::swap( mData, other.mData );
    }

    class Iterator {
    public:
        HOST_DEVICE Iterator( Type* ptr ): ptr( ptr ) {}

        HOST_DEVICE Iterator& operator++() {
            ++ptr;
            return *this;
        }

        HOST_DEVICE Iterator& operator+=( size_t n ) {
            ptr += n;
            return *this;
        }

        HOST_DEVICE Iterator& operator--() {
            --ptr;
            return *this;
        }

        HOST_DEVICE Iterator& operator-=( size_t n ) {
            ptr -= n;
            return *this;
        }

        HOST_DEVICE Type& operator*() {
            return *ptr;
        }

        HOST_DEVICE const Type& operator*() const {
            return *ptr;
        }

        HOST_DEVICE Type* operator->() {
            return ptr;
        }

        HOST_DEVICE const Type* operator->() const {
            return ptr;
        }

        HOST_DEVICE int operator-( const Iterator& other ) const {
            return ptr - other.ptr;
        }

        HOST_DEVICE bool operator==( const Iterator& other ) const {
            return ptr == other.ptr;
        }

        HOST_DEVICE bool operator!=( const Iterator& other ) const {
            return ptr != other.ptr;
        }

        HOST_DEVICE bool operator>=( const Iterator& other ) const {
            return ptr >= other.ptr;
        }

        HOST_DEVICE bool operator<=( const Iterator& other ) const {
            return ptr < other.ptr;
        }

        HOST_DEVICE bool operator>( const Iterator& other ) const {
            return ptr > other.ptr;
        }

        HOST_DEVICE bool operator<( const Iterator& other ) const {
            return ptr < other.ptr;
        }

    private:
        Type* ptr;
    };

    HOST_DEVICE Iterator begin() {
        return Iterator( mData );
    }

    HOST_DEVICE const Iterator begin() const {
        return Iterator( mData );
    }

    HOST_DEVICE Iterator end() {
        return Iterator( mData + mSize );
    }

    HOST_DEVICE const Iterator end() const {
        return Iterator( mData + mSize );
    }

#if HIP_ENABLED

    HOST Vector* copyToDevice() {
        auto device = HIP::allocateOnDevice<Vector>();
        auto data = HIP::allocateOnDevice<Type>( mCap );

        if constexpr ( isPointer<Type>::value ) {
            Type tempData[mSize];
            for ( int i = 0; i < mSize; ++i )
                tempData[i] = mData[i]->copyToDevice();
            HIP::copyToDevice( tempData, data, mSize );
        } else
            HIP::copyToDevice( mData, data, mSize );

        Type* originalData = mData;
        mData = data;
        HIP::copyToDevice( this, device );
        mData = originalData;
        return device;
    }

    HOST Vector* copyToHost() {
        auto host = new Vector();
        HIP::copyToHost( host, this );
        Type hostData[mSize];

        if constexpr ( isPointer<Type>::value ) {
            for ( int i = 0; i < mSize; ++i )
                hostData[i] = mData[i]->copyToHost();
        } else {
            HIP::copyToHost( hostData, mData, mSize );
        }
        host->mData = hostData;
        return host;
    }

    HOST void deallocateOnDevice() {
        if constexpr ( isPointer<Type>::value ) {
            for ( int i = 0; i < mSize; ++i ) {
                mData[i]->deallocateOnDevice();
            }
        } else {
            HIP::deallocateOnDevice( mData );
        }

        HIP::deallocateOnDevice<Vector>( this );
    }
#endif

private:
    Allocator mAlloc;
    Type* mData;
    size_t mSize;
    size_t mCap;
};


#endif //VECTOR_VECTOR_H

