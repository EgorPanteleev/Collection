//
// Created by auser on 10/20/24.
//

#ifndef MATH_VEC2_H
#define MATH_VEC2_H
#include <iostream>
#include <optional>
#include <cmath>

template<typename Type>
class Vec2 {
public:
    HOST_DEVICE Vec2() = default;

    HOST_DEVICE Vec2(const Type& num ) {
        data[0] = num;
        data[1] = num;
    }

    HOST_DEVICE Vec2(Type x, Type y ) {
        data[0] = x;
        data[1] = y;
    }

    HOST_DEVICE Vec2(const Vec2<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
    }

    HOST_DEVICE ~Vec2() = default;

    HOST_DEVICE Vec2<Type>& operator=(const Vec2<Type>& other ) {
        if ( this == &other ) return *this;
        data[0] = other.data[0];
        data[1] = other.data[1];
        return *this;
    }

    HOST_DEVICE Vec2<Type>& operator+=(const Vec2<Type>& other ) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        return *this;
    }

    HOST_DEVICE Vec2<Type>& operator+=(Type a ) {
        data[0] += a;
        data[1] += a;
        return *this;
    }

    HOST_DEVICE Vec2<Type> operator+(const Vec2<Type>& other ) const {
        return { data[0] + other.data[0], data[1] + other.data[1] };
    }

    HOST_DEVICE Vec2<Type>& operator-=(const Vec2<Type>& other ) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        return *this;
    }

    HOST_DEVICE Vec2<Type>& operator-=(Type a ) {
        data[0] -= a;
        data[1] -= a;
        return *this;
    }

    HOST_DEVICE Vec2<Type> operator-(const Vec2<Type>& other ) const {
        return { data[0] - other.data[0], data[1] - other.data[1] };
    }

    HOST_DEVICE Vec2<Type>& operator*=(const Vec2<Type>& other ) {
        data[0] *= other.data[0];
        data[1] *= other.data[1];
        return *this;
    }

    HOST_DEVICE Vec2<Type>& operator*=(Type a ) {
        data[0] *= a;
        data[1] *= a;
        return *this;
    }

    HOST_DEVICE Vec2<Type> operator*(const Vec2<Type>& other ) const {
        return { data[0] * other.data[0], data[1] * other.data[1] };
    }

    HOST_DEVICE Vec2<Type>& operator/=(const Vec2<Type>& other ) {
        data[0] /= other.data[0];
        data[1] /= other.data[1];
        return *this;
    }

    HOST_DEVICE Vec2<Type>& operator/=(Type a ) {
        data[0] /= a;
        data[1] /= a;
        return *this;
    }

    HOST_DEVICE Vec2<Type> operator/(const Vec2<Type>& other ) const {
        return { data[0] / other.data[0], data[1] / other.data[1] };
    }

    HOST_DEVICE bool operator==( const Vec2<Type>& other ) const {
        return data[0] == other.data[0] && data[1] == other.data[1];
    }

    HOST_DEVICE bool operator!=( const Vec2<Type>& other ) const {
        return !( *this == other );
    }

    HOST_DEVICE Vec2<Type> normalize() const {
        Type len = sqrt( pow( data[0], 2 ) +  pow( data[1], 2 ));
        if ( len == 0 ) return *this;
        return *this / len;
    }

    HOST_DEVICE Type& operator[]( int index ) {
        if ( index == 0 ) return data[0];
        if ( index == 1 ) return data[1];
        return (Type &) std::nullopt;
    }

    HOST_DEVICE const Type& operator[]( int index ) const {
        if ( index == 0 ) return data[0];
        if ( index == 1 ) return data[1];
        return (const Type &) std::nullopt;
    }

public:
    Type data[2];
};
template<typename Type>
HOST_DEVICE inline std::ostream& operator << (std::ostream &os, const Vec2<Type> &vec ) {
    return os << "( " << vec.data[0] << ", " << vec.data[1] << " )" << "\n";
}

template<typename Type>
HOST_DEVICE inline Vec2<Type> operator+(Type a, const Vec2<Type>& vec ) {
    return { vec.data[0] + a, vec.data[1] + a };
}

template<typename Type>
HOST_DEVICE inline Vec2<Type> operator+(const Vec2<Type>& vec, Type a ) {
    return { vec.data[0] + a, vec.data[1] + a };
}

template<typename Type>
HOST_DEVICE inline Vec2<Type> operator*(Type a, const Vec2<Type>& vec ) {
    return { vec.data[0] * a, vec.data[1] * a };
}

template<typename Type>
HOST_DEVICE inline Vec2<Type> operator*(const Vec2<Type>& vec, Type a ) {
    return { vec.data[0] * a, vec.data[1] * a };
}

template<typename Type>
HOST_DEVICE inline Vec2<Type> operator/(Type a, const Vec2<Type>& vec ) {
    return { a / vec.data[0], a / vec.data[1] };
}

template<typename Type>
HOST_DEVICE inline Vec2<Type> operator/(const Vec2<Type>& vec, Type a ) {
    return { vec.data[0] / a, vec.data[1] / a };
}

using Vec2d = Vec2<double>;
using Vec2f = Vec2<double>;
using Vec2i = Vec2<int>;
using Point2d = Vec2<double>;
using Point2f = Vec2<double>;
using Point2i = Vec2<int>;

#endif //MATH_VEC2_H
