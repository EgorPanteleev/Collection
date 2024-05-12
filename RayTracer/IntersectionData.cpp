//
// Created by auser on 5/12/24.
//

#include "IntersectionData.h"
#include <limits>

IntersectionData::IntersectionData(): t( std::numeric_limits<float>::max() ), N(), object( nullptr ) {};

IntersectionData::IntersectionData( float t, const Vector3f& N, Object* obj ): t( t ), N( N ), object( obj ) {};