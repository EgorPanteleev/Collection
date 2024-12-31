//
// Created by auser on 11/26/24.
//
#pragma once

#include <limits>
#include "Random.h"
#include "Vec3.h"
#include "OutputStream.h"

constexpr double EPS = 0.0001;

constexpr double INF = std::numeric_limits<double>::infinity();

extern OutputStream<std::ostream> MESSAGE;

template <typename Type>
Type toRadians( Type degrees ) {
    return degrees * ( M_PI / 180 );
}

template <typename Type>
Type toDegrees( Type radians ) {
    return radians * ( 180 / M_PI );
}


