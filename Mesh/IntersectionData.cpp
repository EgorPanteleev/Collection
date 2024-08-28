//
// Created by auser on 5/12/24.
//

#include "IntersectionData.h"
#include <limits>

IntersectionData::IntersectionData():
        t( __FLT_MAX__ ), primitive( nullptr ) {};

IntersectionData::IntersectionData( float t, Primitive* prim ):
        t( t ), primitive( prim ) {};