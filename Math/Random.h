//
// Created by auser on 10/20/24.
//

#ifndef COLLECTION_RANDOM_H
#define COLLECTION_RANDOM_H

#endif //COLLECTION_RANDOM_H
#include <random>
#include <limits>

inline double randomDouble() {
    return std::rand() / ( RAND_MAX + 1.0 );
}

inline double randomDouble( double min, double max ) {
    return min + ( max - min ) * randomDouble();
}