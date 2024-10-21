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
    Vec4(): data() {}

    Vec4(const Type& num ) {
        data[0] = num;
        data[1] = num;
        data[2] = num;
        data[3] = num;
    }

    Vec4(Type x, Type y, Type z, Type w ) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }

    Vec4(const Vec4<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        data[3] = other.data[3];
    }
    
    ~Vec4() {
    }

    Vec4<Type>& operator=(const Vec4<Type>& other ) {
        if ( this == &other ) return *this;
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        data[3] = other.data[3];
        return *this;
    }

    Vec4<Type>& operator+=(const Vec4<Type>& other ) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        data[2] += other.data[2];
        data[3] += other.data[3];
        return *this;
    }

    Vec4<Type>& operator+=(Type a ) {
        data[0] += a;
        data[1] += a;
        data[2] += a;
        data[3] += a;
        return *this;
    }

    Vec4<Type> operator+(const Vec4<Type>& other ) const {
        return { data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2], data[3] + other.data[3] };
    }

    Vec4<Type>& operator-=(const Vec4<Type>& other ) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        data[2] -= other.data[2];
        data[3] -= other.data[3];
        return *this;
    }

    Vec4<Type>& operator-=(Type a ) {
        data[0] -= a;
        data[1] -= a;
        data[2] -= a;
        data[3] -= a;
        return *this;
    }

    Vec4<Type> operator-(const Vec4<Type>& other ) const {
        return { data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2] };
    }

    Vec4<Type>& operator*=(const Vec4<Type>& other ) {
        data[0] *= other.data[0];
        data[1] *= other.data[1];
        data[2] *= other.data[2];
        data[3] *= other.data[3];
        return *this;
    }

    Vec4<Type>& operator*=(Type a ) {
        data[0] *= a;
        data[1] *= a;
        data[2] *= a;
        data[3] *= a;
        return *this;
    }

    Vec4<Type> operator*(const Vec4<Type>& other ) const {
        return { data[0] * other.data[0], data[1] * other.data[1], data[2] * other.data[2], data[3] * other.data[3] };
    }

    Vec4<Type>& operator/=(const Vec4<Type>& other ) {
        data[0] /= other.data[0];
        data[1] /= other.data[1];
        data[2] /= other.data[2];
        data[3] /= other.data[3];
        return *this;
    }

    Vec4<Type>& operator/=(Type a ) {
        data[0] /= a;
        data[1] /= a;
        data[2] /= a;
        data[3] /= a;
        return *this;
    }

    Vec4<Type> operator/(const Vec4<Type>& other ) const {
        return { data[0] / other.data[0], data[1] / other.data[1], data[2] / other.data[2], data[3] / other.data[3] };
    }

    bool operator==( const Vec4<Type>& other ) const {
        return data[0] == other.data[0] && data[1] == other.data[1] && data[2] == other.data[2] && data[3] == other.data[3];
    }

    bool operator!=( const Vec4<Type>& other ) const {
        return !( *this == other );
    }

    Vec4<Type> normalize() const {
        Type len = sqrt( pow( data[0], 2 ) +  pow( data[1], 2 ) + pow( data[2], 2 ));
        if ( len == 0 ) return *this;
        return *this / len;
    }

    Type& operator[]( int index ) {
        return data[index];
    }

    const Type& operator[]( int index ) const {
        return data[index];
    }

public:
    Type data[4];
};
template<typename Type>
std::ostream& operator << (std::ostream &os, const Vec4<Type> &vec ) {
    return os << "( " << vec.data[0] << ", " << vec.data[1] << ", " << vec.data[2] << ", " << vec.data[3] << " )" << "\n";
}

template<typename Type>
Vec4<Type> operator+(Type a, const Vec4<Type>& vec ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a, vec.data[3] + a };
}

template<typename Type>
Vec4<Type> operator+(const Vec4<Type>& vec, Type a ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a, vec.data[3] + a };
}

template<typename Type>
Vec4<Type> operator*(Type a, const Vec4<Type>& vec ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a, vec.data[3] * a };
}

template<typename Type>
Vec4<Type> operator*(const Vec4<Type>& vec, Type a ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a, vec.data[3] * a };
}

template<typename Type>
Vec4<Type> operator/(Type a, const Vec4<Type>& vec ) {
    return { a / vec.data[0], a / vec.data[1], a / vec.data[2], a / vec.data[3] };
}

template<typename Type>
Vec4<Type> operator/(const Vec4<Type>& vec, Type a ) {
    return { vec.data[0] / a, vec.data[1] / a, vec.data[2] / a, vec.data[3] / a };
}

typedef Vec4<double> Vec4d;
typedef Vec4<double> Vec4f;
typedef Vec4<int> Vec4i;

#endif //MATH_VEC4_H
