//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_INTERVAL_H
#define COLLECTION_INTERVAL_H
#include <limits>
#include "SystemUtils.h"
template <typename Type>
class Interval {
public:
    HOST_DEVICE Interval(): min( -std::numeric_limits<Type>::infinity() ), max( std::numeric_limits<Type>::infinity() ) {}
    HOST_DEVICE Interval( Type min, Type max ): min( min ), max( max ) {}

    HOST_DEVICE Type size() const {
        return max - min;
    }

    HOST_DEVICE bool contains( Type x ) const {
        return x >= min && x <= max;
    }

    HOST_DEVICE bool surrounds( Type x ) const {
        return x > min && x < max;
    }

    HOST_DEVICE Type clamp( Type x ) const {
        if ( x < min ) return min;
        if ( x > max ) return max;
        return x;
    }

    Type min;
    Type max;
};


using IntervalI = Interval<int>;
using IntervalF = Interval<float>;
using IntervalD = Interval<double>;


#endif //COLLECTION_INTERVAL_H
