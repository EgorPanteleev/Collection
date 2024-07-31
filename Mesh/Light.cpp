//
// Created by igor on 14.01.2024.
//

#include "Light.h"
#include "Color.h"

__host__ __device__ Light::Light():intensity( 0 ) {
    lightColor = RGB(255,255,255);
}


__host__ __device__ Light::Type Light::getType() const {
    return BASE;
}