//
// Created by igor on 08.01.2024.
//

#include "Cube.h"

Cube::Cube(): p1(), p2() {
}
Cube::Cube( const Vector3f& _p1, const Vector3f& _p2): p1(_p1), p2(_p2) {
}

Cube::Cube( float x1, float y1, float z1, float x2, float y2, float z2):  p1( Vector3f( x1, y1, z1) ),
                                                                                p2( Vector3f( x2, y2, z2) ){ }

bool Cube::isContainPoint( const Vector3f& p ) const {

}

float Cube::intersectsWithRay( const Ray& ray ) const {

}

Vector3f Cube::getNormal( const Vector3f& p ) const {
    return {};
}