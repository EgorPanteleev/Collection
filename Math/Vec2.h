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
    Vec2(): data() {}

    Vec2(const Type& num ) {
        data[0] = num;
        data[1] = num;
    }

    Vec2(Type x, Type y ) {
        data[0] = x;
        data[1] = y;
    }

    Vec2(const Vec2<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
    }

    ~Vec2() {
    }

    Vec2<Type>& operator=(const Vec2<Type>& other ) {
        if ( this == &other ) return *this;
        data[0] = other.data[0];
        data[1] = other.data[1];
        return *this;
    }

    Vec2<Type>& operator+=(const Vec2<Type>& other ) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        return *this;
    }

    Vec2<Type>& operator+=(Type a ) {
        data[0] += a;
        data[1] += a;
        return *this;
    }

    Vec2<Type> operator+(const Vec2<Type>& other ) const {
        return { data[0] + other.data[0], data[1] + other.data[1] };
    }

    Vec2<Type>& operator-=(const Vec2<Type>& other ) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        return *this;
    }

    Vec2<Type>& operator-=(Type a ) {
        data[0] -= a;
        data[1] -= a;
        return *this;
    }

    Vec2<Type> operator-(const Vec2<Type>& other ) const {
        return { data[0] - other.data[0], data[1] - other.data[1] };
    }

    Vec2<Type>& operator*=(const Vec2<Type>& other ) {
        data[0] *= other.data[0];
        data[1] *= other.data[1];
        return *this;
    }

    Vec2<Type>& operator*=(Type a ) {
        data[0] *= a;
        data[1] *= a;
        return *this;
    }

    Vec2<Type> operator*(const Vec2<Type>& other ) const {
        return { data[0] * other.data[0], data[1] * other.data[1] };
    }

    Vec2<Type>& operator/=(const Vec2<Type>& other ) {
        data[0] /= other.data[0];
        data[1] /= other.data[1];
        return *this;
    }

    Vec2<Type>& operator/=(Type a ) {
        data[0] /= a;
        data[1] /= a;
        return *this;
    }

    Vec2<Type> operator/(const Vec2<Type>& other ) const {
        return { data[0] / other.data[0], data[1] / other.data[1] };
    }

    bool operator==( const Vec2<Type>& other ) const {
        return data[0] == other.data[0] && data[1] == other.data[1];
    }

    bool operator!=( const Vec2<Type>& other ) const {
        return !( *this == other );
    }

    Vec2<Type> normalize() const {
        Type len = sqrt( pow( data[0], 2 ) +  pow( data[1], 2 ));
        if ( len == 0 ) return *this;
        return *this / len;
    }

    Type& operator[]( int index ) {
        if ( index == 0 ) return data[0];
        if ( index == 1 ) return data[1];
        return (Type &) std::nullopt;
    }

    const Type& operator[]( int index ) const {
        if ( index == 0 ) return data[0];
        if ( index == 1 ) return data[1];
        return (const Type &) std::nullopt;
    }

public:
    Type data[2];
};
template<typename Type>
inline std::ostream& operator << (std::ostream &os, const Vec2<Type> &vec ) {
    return os << "( " << vec.data[0] << ", " << vec.data[1] << " )" << "\n";
}

template<typename Type>
inline Vec2<Type> operator+(Type a, const Vec2<Type>& vec ) {
    return { vec.data[0] + a, vec.data[1] + a };
}

template<typename Type>
inline Vec2<Type> operator+(const Vec2<Type>& vec, Type a ) {
    return { vec.data[0] + a, vec.data[1] + a };
}

template<typename Type>
inline Vec2<Type> operator*(Type a, const Vec2<Type>& vec ) {
    return { vec.data[0] * a, vec.data[1] * a };
}

template<typename Type>
inline Vec2<Type> operator*(const Vec2<Type>& vec, Type a ) {
    return { vec.data[0] * a, vec.data[1] * a };
}

template<typename Type>
inline Vec2<Type> operator/(Type a, const Vec2<Type>& vec ) {
    return { a / vec.data[0], a / vec.data[1] };
}

template<typename Type>
inline Vec2<Type> operator/(const Vec2<Type>& vec, Type a ) {
    return { vec.data[0] / a, vec.data[1] / a };
}

using Vec2d = Vec2<double>;
using Vec2f = Vec2<double>;
using Vec2i = Vec2<int>;
using Point2d = Vec2<double>;
using Point2f = Vec2<double>;
using Point2i = Vec2<int>;

#endif //MATH_VEC2_H
