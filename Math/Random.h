//
// Created by auser on 10/20/24.
//

#ifndef COLLECTION_RANDOM_H
#define COLLECTION_RANDOM_H
#include <random>
#include <limits>
#include <hiprand/hiprand_kernel.h>
//#include <rocrand_kernel.h>
#include "SystemUtils.h"


HOST inline double randomDouble() {
    return std::rand() / ( RAND_MAX + 1.0 );
}

HOST inline double randomDouble( double min, double max ) {
    return min + ( max - min ) * randomDouble();
}

#ifdef HIP_ENABLED

DEVICE inline double randomDouble( hiprandState& state ) {
    return hiprand_uniform_double(&state);
}

DEVICE inline double randomDouble( double min, double max, hiprandState& state ) {
    return min + ( max - min ) * randomDouble( state );
}

#endif

#endif //COLLECTION_RANDOM_H

