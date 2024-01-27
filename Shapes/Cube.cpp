//
// Created by igor on 08.01.2024.
//

#include "Cube.h"

Cube::Cube(): p1(), p2() {
}
Cube::Cube( Vector3f _p1, Vector3f _p2): p1(_p1), p2(_p2) {
}

Cube::Cube( double x1, double y1, double z1, double x2, double y2, double z2):  p1( Vector3f( x1, y1, z1) ),
                                                                                p2( Vector3f( x2, y2, z2) ){ }

bool Cube::isContainPoint( Vector3f p ) const {

}

double Cube::intersectsWithRay( const Ray& ray ) {

}