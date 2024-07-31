//
// Created by auser on 5/12/24.
//

#include "IntersectionData.h"
#include <limits>

__host__ __device__ IntersectionData::IntersectionData(): t( std::numeric_limits<float>::max() ), N(), triangle() {};

__host__ __device__ IntersectionData::IntersectionData( float t, const Vector3f& N, Triangle* tr ): t( t ), N( N ), triangle( tr ) {};