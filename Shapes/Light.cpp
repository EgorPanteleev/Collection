//
// Created by igor on 14.01.2024.
//

#include "Light.h"

Light::Light():origin(), intensity(){}

Light::Light( const Vector3f& origin, float intensity ): origin( origin ), intensity( intensity ) {}