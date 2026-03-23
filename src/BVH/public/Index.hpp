//
// Created by igor on 11/1/25.
//

#ifndef INDEX_HPP
#define INDEX_HPP

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <climits>

namespace crv::graphics {
    template <size_t bits>
    struct UnsignedInt {};

    template<> struct UnsignedInt<8 > { using Type = uint8_t ; };
    template<> struct UnsignedInt<16> { using Type = uint16_t; };
    template<> struct UnsignedInt<32> { using Type = uint32_t; };
    template<> struct UnsignedInt<64> { using Type = uint64_t; };

    template<size_t bits>
    using UnsignedIntType = typename UnsignedInt<bits>::Type;

    template <typename Type>
    constexpr Type makeBitMask(size_t bits) {
        assert(sizeof(Type) * CHAR_BIT > bits);
        return (static_cast<Type>(1) << bits) - 1;
    }

    template <size_t bits, size_t primCountBits>
    class Index {
    public:
        using Type = UnsignedIntType<bits>;
        Index() = default;
        explicit Index(Type value): mValue(value) {}
        explicit Index(size_t firstPrim, size_t primCount) { set(firstPrim, primCount); }
        explicit Index(size_t firstChild) { set(firstChild, 0); }
        Type id() const { return mValue >> primCountBits; }
        Type primCount() const { return mValue & MAX_PRIM; }
        bool isLeaf() const { return primCount() != 0; }
        bool isInner() const { return !isLeaf(); }
        void setID(size_t id) { set(id, static_cast<size_t>(primCount())); }
        void setPrimCount(size_t primCount) { set(static_cast<size_t>(id()), primCount); }

        static Type maxPrim() { return MAX_PRIM; }
    private:
        static constexpr Type MAX_ID = makeBitMask<Type>(bits - primCountBits);
        static constexpr Type MAX_PRIM = makeBitMask<Type>(primCountBits);

        void set(size_t id, size_t primCount) { mValue = static_cast<Type>(id) << primCountBits |
                                                        (static_cast<Type>(primCount) & MAX_PRIM);
            assert(id <= static_cast<size_t>(MAX_ID));
            assert(primCount <= static_cast<size_t>(MAX_PRIM));
        }

        Type mValue;
    };
}

#endif //INDEX_HPP
