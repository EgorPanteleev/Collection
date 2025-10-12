//
// Created by auser on 10/20/24.
//

#ifndef MATH_VEC4_H
#define MATH_VEC4_H
#include <iostream>
#include <optional>
#include <cmath>

template<typename Type>
class Vec4 {
public:
    HOST_DEVICE Vec4() = default;

    HOST_DEVICE Vec4(const Type& num ) {
        data[0] = num;
        data[1] = num;
        data[2] = num;
        data[3] = num;
    }

    HOST_DEVICE Vec4(Type x, Type y, Type z, Type w ) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }

    HOST_DEVICE Vec4(const Vec3<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        data[3] = 1;
    }

    HOST_DEVICE Vec4(const Vec4<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        data[3] = other.data[3];
    }

    HOST_DEVICE ~Vec4() = default;

    HOST_DEVICE Vec4<Type>& operator=(const Vec4<Type>& other ) {
        if ( this == &other ) return *this;
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        data[3] = other.data[3];
        return *this;
    }

    HOST_DEVICE Vec4<Type>& operator+=(const Vec4<Type>& other ) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        data[2] += other.data[2];
        data[3] += other.data[3];
        return *this;
    }

    HOST_DEVICE Vec4<Type>& operator+=(Type a ) {
        data[0] += a;
        data[1] += a;
        data[2] += a;
        data[3] += a;
        return *this;
    }

    HOST_DEVICE Vec4<Type> operator+(const Vec4<Type>& other ) const {
        return { data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2], data[3] + other.data[3] };
    }

    HOST_DEVICE Vec4<Type>& operator-=(const Vec4<Type>& other ) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        data[2] -= other.data[2];
        data[3] -= other.data[3];
        return *this;
    }

    HOST_DEVICE Vec4<Type>& operator-=(Type a ) {
        data[0] -= a;
        data[1] -= a;
        data[2] -= a;
        data[3] -= a;
        return *this;
    }

    HOST_DEVICE Vec4<Type> operator-(const Vec4<Type>& other ) const {
        return { data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2] };
    }

    HOST_DEVICE Vec4<Type>& operator*=(const Vec4<Type>& other ) {
        data[0] *= other.data[0];
        data[1] *= other.data[1];
        data[2] *= other.data[2];
        data[3] *= other.data[3];
        return *this;
    }

    HOST_DEVICE Vec4<Type>& operator*=(Type a ) {
        data[0] *= a;
        data[1] *= a;
        data[2] *= a;
        data[3] *= a;
        return *this;
    }

    HOST_DEVICE Vec4<Type> operator*(const Vec4<Type>& other ) const {
        return { data[0] * other.data[0], data[1] * other.data[1], data[2] * other.data[2], data[3] * other.data[3] };
    }

    HOST_DEVICE Vec4<Type>& operator/=(const Vec4<Type>& other ) {
        data[0] /= other.data[0];
        data[1] /= other.data[1];
        data[2] /= other.data[2];
        data[3] /= other.data[3];
        return *this;
    }

    HOST_DEVICE Vec4<Type>& operator/=(Type a ) {
        data[0] /= a;
        data[1] /= a;
        data[2] /= a;
        data[3] /= a;
        return *this;
    }

    HOST_DEVICE Vec4<Type> operator/(const Vec4<Type>& other ) const {
        return { data[0] / other.data[0], data[1] / other.data[1], data[2] / other.data[2], data[3] / other.data[3] };
    }

    HOST_DEVICE bool operator==( const Vec4<Type>& other ) const {
        return data[0] == other.data[0] && data[1] == other.data[1] && data[2] == other.data[2] && data[3] == other.data[3];
    }

    HOST_DEVICE bool operator!=( const Vec4<Type>& other ) const {
        return !( *this == other );
    }

    HOST_DEVICE Vec4<Type> normalize() const {
        Type len = sqrt( pow( data[0], 2 ) +  pow( data[1], 2 ) + pow( data[2], 2 ));
        if ( len == 0 ) return *this;
        return *this / len;
    }

    HOST_DEVICE Type& operator[]( int index ) {
        return data[index];
    }

    HOST_DEVICE const Type& operator[]( int index ) const {
        return data[index];
    }

public:
    Type data[4];
};
template<typename Type>
HOST_DEVICE inline std::ostream& operator << (std::ostream &os, const Vec4<Type> &vec ) {
    return os << "( " << vec.data[0] << ", " << vec.data[1] << ", " << vec.data[2] << ", " << vec.data[3] << " )" << "\n";
}

template<typename Type>
HOST_DEVICE inline Vec4<Type> operator+(Type a, const Vec4<Type>& vec ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a, vec.data[3] + a };
}

template<typename Type>
HOST_DEVICE inline Vec4<Type> operator+(const Vec4<Type>& vec, Type a ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a, vec.data[3] + a };
}

template<typename Type>
HOST_DEVICE inline Vec4<Type> operator*(Type a, const Vec4<Type>& vec ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a, vec.data[3] * a };
}

template<typename Type>
HOST_DEVICE inline Vec4<Type> operator*(const Vec4<Type>& vec, Type a ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a, vec.data[3] * a };
}

template<typename Type>
HOST_DEVICE inline Vec4<Type> operator/(Type a, const Vec4<Type>& vec ) {
    return { a / vec.data[0], a / vec.data[1], a / vec.data[2], a / vec.data[3] };
}

template<typename Type>
HOST_DEVICE inline Vec4<Type> operator/(const Vec4<Type>& vec, Type a ) {
    return { vec.data[0] / a, vec.data[1] / a, vec.data[2] / a, vec.data[3] / a };
}

using Vec4d = Vec4<double>;
using Vec4f = Vec4<double>;
using Vec4i = Vec4<int>;
using Point4d = Vec4<double>;
using Point4f = Vec4<double>;
using Point4i = Vec4<int>;

#endif //MATH_VEC4_H
