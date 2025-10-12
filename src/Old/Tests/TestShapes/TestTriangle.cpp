#include <iostream>
#include "Triangle.h"
#include <cmath>
Triangle createTriangle1() {
    return { Vec3d(0,0,0), Vec3d(5,2,0), Vec3d(1, 7, 0) };
}

int testContainPoint() {
    Triangle tr1 = createTriangle1();
    if ( !tr1.isContainPoint( {2, 2, 0})) return 1;
    if ( tr1.isContainPoint( {2, 2, 0.1})) return 1;
    if ( !tr1.isContainPoint( {2, 0.9, 0})) return 1;
    if ( !tr1.isContainPoint( {3, 1.2, 0})) return 1;
    if ( !tr1.isContainPoint( {1, 3, 0})) return 1;
    if ( tr1.isContainPoint( {1, 0.2, 0})) return 1;
    if ( tr1.isContainPoint( {1, 7.05, 0})) return 1;
    if ( !tr1.isContainPoint( {5, 2, 0})) return 1;
    if ( tr1.isContainPoint( {5, 2.05, 0})) return 1;
    if ( !tr1.isContainPoint( {4, 2, 0})) return 1;
    if ( !tr1.isContainPoint( {3, 4.3, 0})) return 1;
    return 0;
}

int testintersectsWithRay() {
    Triangle tr1 = createTriangle1();
    if ( tr1.intersectsWithRay( Ray({0,0,-1}, {0,0,1}) ) != 1 ) return 1;
    if ( tr1.intersectsWithRay( Ray({0,0,-2}, {0,0,1}) ) != 2 ) return 1;
    if ( tr1.intersectsWithRay( Ray({0,0,-999}, {0,0,1}) ) != 999 ) return 1;
    if ( (int) ( tr1.intersectsWithRay( Ray({0,0,-1}, {2,3,1}) ) * 1000 )
    != (int) ( sqrtf(14) * 1000 ) ) return 1;
    if (  tr1.intersectsWithRay( Ray({0,0,-1}, {1,0,0}) ) != __FLT_MAX__ ) return 1;
    return 0;
}

int testGetNormal() {
    Triangle tr1 = createTriangle1();
    if ( tr1.getNormal() != Vec3d( 0,0,1 ) ) return 1;
    return 0;
}

int main() {
    int res = 0;
    res += testContainPoint();
    res += testintersectsWithRay();
    res += testGetNormal();
    return res;
}