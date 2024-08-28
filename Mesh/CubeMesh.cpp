//
// Created by auser on 5/12/24.
//

#include "CubeMesh.h"
#include "IntersectionData.h"
CubeMesh::CubeMesh(): p1({0,0,0}), p2({1,1,1}) {
    fillTriangles();
}
CubeMesh::CubeMesh( const Vector3f& _p1, const Vector3f& _p2): p1(_p1), p2(_p2) {
    fillTriangles();
}

CubeMesh::CubeMesh( const Vector3f& _p1, const Vector3f& _p2, const Material& _material ): p1(_p1), p2(_p2) {
    material = _material;
    fillTriangles();
}

void CubeMesh::fillTriangles() {
    Vector3f f1 = p1;
    Vector3f f2 = { p2.getX(), p1.getY(), p1.getZ() };
    Vector3f f3 = { p2.getX(), p1.getY(), p2.getZ() };
    Vector3f f4 = { p1.getX(), p1.getY(), p2.getZ() };

    Vector3f b1 = { p1.getX(), p2.getY(), p1.getZ() };
    Vector3f b2 = { p2.getX(), p2.getY(), p1.getZ() };
    Vector3f b3 = p2;
    Vector3f b4 = { p1.getX(), p2.getY(), p2.getZ() };
    // down
    primitives.push_back( new Triangle{ f1, f2, f3 } );
    primitives.push_back( new Triangle{ f1, f3, f4 } );
    //up
    primitives.push_back( new Triangle{ b1, b3, b2 } );
    primitives.push_back( new Triangle{ b1, b4, b3 } );
    //left
    primitives.push_back( new Triangle{ b1, f1, f4 } );
    primitives.push_back( new Triangle{ b1, f4, b4 } );
    //right
    primitives.push_back( new Triangle{ f2, b2, f3 } );
    primitives.push_back( new Triangle{ f3, b2, b3 } );
    //front
    primitives.push_back( new Triangle{ f2, f1, b1 } );
    primitives.push_back( new Triangle{ f2, b1, b2 } );
    //back
    primitives.push_back( new Triangle{ f4, f3, b4 } );
    primitives.push_back( new Triangle{ f3, b3, b4 } );
    for ( auto primitive: primitives )
        primitive->setMaterial( material );
}



