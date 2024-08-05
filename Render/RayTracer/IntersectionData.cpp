//
// Created by auser on 5/12/24.
//

#include "IntersectionData.h"
#include <limits>

IntersectionData::IntersectionData():
t( __FLT_MAX__ ), N(), triangle( nullptr ), sphere( nullptr ) {};

IntersectionData::IntersectionData( float t, const Vector3f& N, Triangle* tr, Sphere* sp ):
                                    t( t ), N( N ), triangle( tr ), sphere( sp ) {};