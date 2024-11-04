//
// Created by auser on 10/20/24.
//

#ifndef MATH_VEC3_H
#define MATH_VEC3_H
#include <iostream>
#include <cmath>

template<typename Type>
class Vec3 {
public:
    Vec3(): data() {}

    Vec3(const Type& num ) {
        data[0] = num;
        data[1] = num;
        data[2] = num;
    }

    Vec3(Type x, Type y, Type z ) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    Vec3(const Vec3<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
    }

    ~Vec3() {
    }

    Vec3<Type>& operator=(const Vec3<Type>& other ) {
        if ( this == &other ) return *this;
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        return *this;
    }

    Vec3<Type>& operator+=(const Vec3<Type>& other ) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        data[2] += other.data[2];
        return *this;
    }

    Vec3<Type>& operator+=(Type a ) {
        data[0] += a;
        data[1] += a;
        data[2] += a;
        return *this;
    }

    Vec3<Type> operator+(const Vec3<Type>& other ) const {
        return { data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2] };
    }

    Vec3<Type>& operator-=(const Vec3<Type>& other ) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        data[2] -= other.data[2];
        return *this;
    }

    Vec3<Type>& operator-=(Type a ) {
        data[0] -= a;
        data[1] -= a;
        data[2] -= a;
        return *this;
    }

    Vec3<Type> operator-(const Vec3<Type>& other ) const {
        return { data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2] };
    }

    Vec3<Type>& operator*=(const Vec3<Type>& other ) {
        data[0] *= other.data[0];
        data[1] *= other.data[1];
        data[2] *= other.data[2];
        return *this;
    }

    Vec3<Type>& operator*=(Type a ) {
        data[0] *= a;
        data[1] *= a;
        data[2] *= a;
        return *this;
    }

    Vec3<Type> operator*(const Vec3<Type>& other ) const {
        return { data[0] * other.data[0], data[1] * other.data[1], data[2] * other.data[2] };
    }

    Vec3<Type>& operator/=(const Vec3<Type>& other ) {
        data[0] /= other.data[0];
        data[1] /= other.data[1];
        data[2] /= other.data[2];
        return *this;
    }

    Vec3<Type>& operator/=(Type a ) {
        data[0] /= a;
        data[1] /= a;
        data[2] /= a;
        return *this;
    }

    Vec3<Type> operator/(const Vec3<Type>& other ) const {
        return { data[0] / other.data[0], data[1] / other.data[1], data[2] / other.data[2] };
    }

    bool operator==( const Vec3<Type>& other ) const {
        return data[0] == other.data[0] && data[1] == other.data[1] && data[2] == other.data[2];
    }

    bool operator!=( const Vec3<Type>& other ) const {
        return !( *this == other );
    }

    Vec3<Type> normalize() const {
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
    Type data[3];
};
template<typename Type>
inline std::ostream& operator << (std::ostream &os, const Vec3<Type> &vec ) {
    return os << "( " << vec.data[0] << ", " << vec.data[1] << ", " << vec.data[2] << " )" << "\n";
}

template<typename Type>
inline Vec3<Type> cross( const Vec3<Type>& first, const Vec3<Type>& second ) {
    return {
            first.data[1] * second.data[2] - first.data[2] * second.data[1],
            first.data[2] * second.data[0] - first.data[0] * second.data[2],
            first.data[0] * second.data[1] - first.data[1] * second.data[0]
    };
}

template<typename Type>
inline double getDistance( const Vec3<Type>& first, const Vec3<Type>& second ) {
    Vec3<Type> tmp = first - second;
    return sqrt( dot( tmp, tmp ) );
}

template<typename Type>
inline Vec3<Type> reflect( const Vec3<Type>& wo, const Vec3<Type>& N ) {
    return  ( wo - N * 2 * dot(N, wo ) ).normalize();
}

template<typename Type>
inline double dot( const Vec3<Type>& first, const Vec3<Type>& second ) {
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

template<typename Type>
inline Vec3<Type> operator+(Type a, const Vec3<Type>& vec ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a };
}

template<typename Type>
inline Vec3<Type> operator+(const Vec3<Type>& vec, Type a ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a };
}

template<typename Type>
inline Vec3<Type> operator*(Type a, const Vec3<Type>& vec ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a };
}

template<typename Type>
inline Vec3<Type> operator*(const Vec3<Type>& vec, Type a ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a };
}

template<typename Type>
inline Vec3<Type> operator/(Type a, const Vec3<Type>& vec ) {
    return { a / vec.data[0], a / vec.data[1], a / vec.data[2] };
}

template<typename Type>
inline Vec3<Type> operator/(const Vec3<Type>& vec, Type a ) {
    return { vec.data[0] / a, vec.data[1] / a, vec.data[2] / a };
}

template<typename Type>
inline Vec3<Type> min( const Vec3<Type>& vec1, const Vec3<Type>& vec2 ) {
    return { std::min( vec1.data[0], vec2.data[0] ),
             std::min( vec1.data[1], vec2.data[1] ),
             std::min( vec1.data[2], vec2.data[2] ) };
}

template<typename Type>
inline Vec3<Type> max( const Vec3<Type>& vec1, const Vec3<Type>& vec2 ) {
    return { std::max( vec1.data[0], vec2.data[0] ),
             std::max( vec1.data[1], vec2.data[1] ),
             std::max( vec1.data[2], vec2.data[2] ) };
}

typedef Vec3<double> Vec3d;
typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;

#endif //MATH_VEC3_H
