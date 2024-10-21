//
// Created by auser on 10/20/24.
//

#include "Utils.h"

double saturate( double z ) {
    if ( z < 0.0f ) return 0.0;
    if ( z > 1.0f ) return 1.0;
    return z;
}

double pow2( double a ) {
    return a * a;
}
