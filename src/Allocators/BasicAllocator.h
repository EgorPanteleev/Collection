//
// Created by auser on 10/31/24.
//

#ifndef COLLECTION_ALLOCATOR_H
#define COLLECTION_ALLOCATOR_H
#include <iostream>
#include "SystemUtils.h"

template <typename Type>
class BasicAllocator {
public:
    HOST_DEVICE Type* allocate( size_t count ) {
        //std::cout << "Allocated " << count * sizeof(Type) << " bytes\n";
        return static_cast<Type*>( ::operator new( count * sizeof(Type) ) );
    }

    HOST_DEVICE void deallocate( Type* ptr, size_t count ) {
        ::operator delete( ptr );
        //std::cout << "Deallocated " << count * sizeof(Type) << " bytes\n";
    }

    template <typename AnotherType, typename... Args>
    HOST_DEVICE void construct( AnotherType* ptr, Args&&... args ) {
        new (ptr) AnotherType( std::forward<Args>( args )... );
        //std::cout << "Constructed\n";
    }

    template <typename AnotherType>
    HOST_DEVICE void destroy( AnotherType* ptr ) {
        ptr->~AnotherType();
        //std::cout << "Destroyed\n";
    }

    using ValueType = Type;

    using Reference = Type&;

    using ConstReference = const Type&;
};


#endif //COLLECTION_ALLOCATOR_H
