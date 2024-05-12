//
// Created by auser on 5/12/24.
//

#include "IntersectionData.h"
#include <limits>

IntersectionData::IntersectionData(): t( std::numeric_limits<float>::max() ), N(), triangle() {};

    IntersectionData::IntersectionData( float t, const Vector3f& N, Triangle* tr ): t( t ), N( N ), triangle( tr ) {};