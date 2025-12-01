//
// Created by igor on 11/1/25.
//

#ifndef INDEX_HPP
#define INDEX_HPP

#include <cstdint>
#include <type_traits>
#include <limits>
#include <cassert>

namespace crv::graphics {
    template <size_t bits>
    struct UnsignedInt {};

    template<> struct UnsignedInt<8 > { using Type = uint8_t ; };
    template<> struct UnsignedInt<16> { using Type = uint16_t; };
    template<> struct UnsignedInt<32> { using Type = uint32_t; };
    template<> struct UnsignedInt<64> { using Type = uint64_t; };

    template<size_t bits>
    using UnsignedIntType = typename UnsignedInt<bits>::Type;

    template <typename Type, std::enable_if_t<std::is_unsigned_v<Type>, bool> = true>
    constexpr Type makeBitMask(size_t bits) { return bits >= std::numeric_limits<Type>::digits ? static_cast<Type>(-1) : (static_cast<Type>(1) << bits) - 1; }

    template <size_t bits, size_t primCountBits>
    class Index {
    public:
        using Type = UnsignedIntType<bits>;
        Index() = default;
        explicit Index(Type value): value(value) {}
        Index(size_t firstPrim, size_t primCount) { set(firstPrim, primCount); }
        Index(size_t firstChild) { set(firstChild, 0); }
        Type id() const { return value >> primCountBits; }
        Type primCount() const { return value & maxPrimCount; }
        bool isLeaf() const { return primCount() != 0; }
        bool isInner() const { return !isLeaf(); }
        void setID(size_t id) { set(id, static_cast<size_t>(primCount())); }
        void setPrimCount(size_t primCount) { set(static_cast<size_t>(id()), primCount); }
    private:
        static constexpr Type maxID = makeBitMask<Type>(bits - primCountBits);
        static constexpr Type maxPrimCount = makeBitMask<Type>(primCountBits);

        void set(size_t id, size_t primCount) { value = static_cast<Type>(id) << primCountBits |
                                                        (static_cast<Type>(primCount) & maxPrimCount);
            assert(id <= static_cast<size_t>(maxID));
            assert(primCount <= static_cast<size_t>(maxPrimCount));
        }

        Type value;
    };
}

#endif //INDEX_HPP
