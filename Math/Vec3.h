//
// Created by auser on 10/20/24.
//

#ifndef MATH_VEC3_H
#define MATH_VEC3_H
#include <iostream>
#include <cmath>
#include "SystemUtils.h"
//
//#if HIP_ENABLED
//    #include "hip/hip_runtime.h"
//#endif

template<typename Type>
class Vec3 {
public:
    HOST_DEVICE Vec3() = default;

    HOST_DEVICE Vec3(const Type& num ) {
        data[0] = num;
        data[1] = num;
        data[2] = num;
    }

    HOST_DEVICE Vec3(Type x, Type y, Type z ) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    HOST_DEVICE Vec3(const Vec3<Type>& other ) {
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
    }

    HOST_DEVICE ~Vec3() = default;

    HOST_DEVICE Vec3<Type>& operator=(const Vec3<Type>& other ) {
        if ( this == &other ) return *this;
        data[0] = other.data[0];
        data[1] = other.data[1];
        data[2] = other.data[2];
        return *this;
    }

    HOST_DEVICE Vec3<Type>& operator+=(const Vec3<Type>& other ) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        data[2] += other.data[2];
        return *this;
    }

    HOST_DEVICE Vec3<Type>& operator+=(Type a ) {
        data[0] += a;
        data[1] += a;
        data[2] += a;
        return *this;
    }

    HOST_DEVICE Vec3<Type> operator+(const Vec3<Type>& other ) const {
        return { data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2] };
    }

    HOST_DEVICE Vec3<Type>& operator-=(const Vec3<Type>& other ) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        data[2] -= other.data[2];
        return *this;
    }

    HOST_DEVICE Vec3<Type>& operator-=(Type a ) {
        data[0] -= a;
        data[1] -= a;
        data[2] -= a;
        return *this;
    }

    HOST_DEVICE Vec3<Type> operator-(const Vec3<Type>& other ) const {
        return { data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2] };
    }

    HOST_DEVICE Vec3<Type> operator-() const {
        return { -data[0], -data[1], -data[2] };
    }

    HOST_DEVICE Vec3<Type>& operator*=(const Vec3<Type>& other ) {
        data[0] *= other.data[0];
        data[1] *= other.data[1];
        data[2] *= other.data[2];
        return *this;
    }

    HOST_DEVICE Vec3<Type>& operator*=(Type a ) {
        data[0] *= a;
        data[1] *= a;
        data[2] *= a;
        return *this;
    }

    HOST_DEVICE Vec3<Type> operator*(const Vec3<Type>& other ) const {
        return { data[0] * other.data[0], data[1] * other.data[1], data[2] * other.data[2] };
    }

    HOST_DEVICE Vec3<Type>& operator/=(const Vec3<Type>& other ) {
        data[0] /= other.data[0];
        data[1] /= other.data[1];
        data[2] /= other.data[2];
        return *this;
    }

    HOST_DEVICE Vec3<Type>& operator/=(Type a ) {
        data[0] /= a;
        data[1] /= a;
        data[2] /= a;
        return *this;
    }

    HOST_DEVICE Vec3<Type> operator/(const Vec3<Type>& other ) const {
        return { data[0] / other.data[0], data[1] / other.data[1], data[2] / other.data[2] };
    }

    HOST_DEVICE bool operator==( const Vec3<Type>& other ) const {
        return data[0] == other.data[0] && data[1] == other.data[1] && data[2] == other.data[2];
    }

    HOST_DEVICE bool operator!=( const Vec3<Type>& other ) const {
        return !( *this == other );
    }

    HOST_DEVICE Type lengthSquared() const {
        return pow( data[0], 2 ) +  pow( data[1], 2 ) + pow( data[2], 2 );
    }

    HOST_DEVICE Type length() const {
        return std::sqrt( lengthSquared() );
    }

    HOST_DEVICE Vec3<Type> normalize() const {
        Type len = length();
        if ( len == 0 ) return *this;
        return *this / len;
    }

    HOST_DEVICE Type& operator[]( int index ) {
        return data[index];
    }

    HOST_DEVICE const Type& operator[]( int index ) const {
        return data[index];
    }

//#if HIP_ENABLED
//    HOST void copyToDevice( Vec3<Type>*& device ) {
//        HIP_ASSERT(hipMalloc( &device, sizeof(Vec3<Type>)));
//        HIP_ASSERT(hipMemcpy(device, this, sizeof(Vec3<Type>), hipMemcpyHostToDevice));
//    }
//
//    HOST void copyToHost( Vec3<Type>* host ) {
//        HIP_ASSERT(hipMemcpy( host, this, sizeof(Vec3<Type>), hipMemcpyDeviceToHost) );
//        HIP_ASSERT(hipFree(this));
//    }
//#endif

public:
    Type data[3];
};
template<typename Type>
HOST_DEVICE inline std::ostream& operator << (std::ostream &os, const Vec3<Type> &vec ) {
    return os << "( " << vec.data[0] << ", " << vec.data[1] << ", " << vec.data[2] << " )" << "\n";
}

template<typename Type>
HOST_DEVICE inline Vec3<Type> min( const Vec3<Type>& vec1, const Vec3<Type>& vec2 ) {
    return { std::min( vec1.data[0], vec2.data[0] ),
             std::min( vec1.data[1], vec2.data[1] ),
             std::min( vec1.data[2], vec2.data[2] ) };
}

template<typename Type>
HOST_DEVICE inline Vec3<Type> max( const Vec3<Type>& vec1, const Vec3<Type>& vec2 ) {
    return { std::max( vec1.data[0], vec2.data[0] ),
             std::max( vec1.data[1], vec2.data[1] ),
             std::max( vec1.data[2], vec2.data[2] ) };
}

template<typename Type>
HOST_DEVICE inline Vec3<Type> cross( const Vec3<Type>& first, const Vec3<Type>& second ) {
    return {
            first.data[1] * second.data[2] - first.data[2] * second.data[1],
            first.data[2] * second.data[0] - first.data[0] * second.data[2],
            first.data[0] * second.data[1] - first.data[1] * second.data[0]
    };
}

template<typename Type>
HOST_DEVICE inline double getDistance( const Vec3<Type>& first, const Vec3<Type>& second ) {
    Vec3<Type> tmp = first - second;
    return sqrt( dot( tmp, tmp ) );
}

template<typename Type>
HOST_DEVICE inline Vec3<Type> reflect( const Vec3<Type>& wo, const Vec3<Type>& N ) {
    return ( wo - N * 2 * dot(N, wo ) ).normalize();
}

template<typename Type>
HOST_DEVICE inline Vec3<Type> refract( const Vec3<Type>& uv, const Vec3<Type>& n, double etaiOverEtat ) {
    auto cosTheta = std::min( dot( -uv, n ), 1.0 );
    Vec3<Type> outPerpendicular = etaiOverEtat * ( uv + cosTheta * n );
    Vec3<Type> outParallel = -std::sqrt( abs( 1.0 - outPerpendicular.lengthSquared() ) ) * n;
    return outPerpendicular + outParallel;
}

template<typename Type>
HOST_DEVICE inline double dot( const Vec3<Type>& first, const Vec3<Type>& second ) {
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

template<typename Type1, typename Type2>
HOST_DEVICE inline Vec3<Type1> operator+(Type2 a, const Vec3<Type1>& vec ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a };
}

template<typename Type1, typename Type2>
HOST_DEVICE inline Vec3<Type1> operator+(const Vec3<Type2>& vec, Type1 a ) {
    return { vec.data[0] + a, vec.data[1] + a, vec.data[2] + a };
}

template<typename Type1, typename Type2>
HOST_DEVICE inline Vec3<Type1> operator*(Type2 a, const Vec3<Type1>& vec ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a };
}

template<typename Type1, typename Type2>
HOST_DEVICE inline Vec3<Type1> operator*(const Vec3<Type1>& vec, Type2 a ) {
    return { vec.data[0] * a, vec.data[1] * a, vec.data[2] * a };
}

template<typename Type1, typename Type2>
HOST_DEVICE inline Vec3<Type1> operator/(Type2 a, const Vec3<Type1>& vec ) {
    return { a / vec.data[0], a / vec.data[1], a / vec.data[2] };
}

template<typename Type1, typename Type2>
HOST_DEVICE inline Vec3<Type1> operator/(const Vec3<Type1>& vec, Type2 a ) {
    return { vec.data[0] / a, vec.data[1] / a, vec.data[2] / a };
}


using Vec3d = Vec3<double>;
using Vec3f = Vec3<double>;
using Vec3i = Vec3<int>;
using Point3d = Vec3<double>;
using Point3f = Vec3<double>;
using Point3i = Vec3<int>;
using RGB = Vec3<double>;
#endif //MATH_VEC3_H
