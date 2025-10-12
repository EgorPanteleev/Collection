#include <iostream>
#include "Vector.h"

int testGet() {
    std::cout<< "Testing getters..."<<std::endl;
    Vec3d vec = { 1.1, 2.2, 3.3 };
    if ( vec.getX() != (double) 1.1 ) return 1;
    if ( vec.getY() != (double) 2.2 ) return 1;
    if ( vec.getZ() != (double) 3.3 ) return 1;
    return 0;
}

int testSet() {
    std::cout<< "Testing setters..."<<std::endl;
    Vec3d vec = { 1.1, 2.2, 3.3 };
    vec.set( {2.3, 2.5, 7.8 } );
    if ( vec.getX() != (double) 2.3 ) return 1;
    if ( vec.getY() != (double) 2.5 ) return 1;
    if ( vec.getZ() != (double) 7.8 ) return 1;
    vec.setX( 5.5 );
    if ( vec.getX() != (double) 5.5 ) return 1;
    vec.setY( 6.9 );
    if ( vec.getY() != (double) 6.9 ) return 1;
    vec.setZ( -12.5 );
    if ( vec.getZ() != (double) -12.5 ) return 1;
    return 0;
}

int testOperators() {
    std::cout<< "Testing operators..."<<std::endl;
    std::cout<< "Testing operator []"<<std::endl;
    Vec3d vec = { 1.1f, 2.2f, 3.3f };
    if ( vec[0] != 1.1f ) return 1;
    if ( vec[1] != 2.2f ) return 1;
    if ( vec[2] != 3.3f ) return 1;
    std::cout<< "Testing operator ="<<std::endl;
    vec[0] = 5.4f;
    if ( vec[0] != 5.4f ) return 1;
    vec[1] = 5.5f;
    if ( vec[1] != 5.5f ) return 1;
    vec[2] = 5.6f;
    if ( vec[2] != 5.6f ) return 1;
    std::cout<< "Testing operator +"<<std::endl;
    Vec3d vec1 = { 0.6f, 1.5f, 2.4f };
    Vec3d vec2 = vec1 + vec;
    if ( vec2[0] != 6.f ) return 1;
    if ( vec2[1] != 7.f ) return 1;
    if ( vec2[2] != 8.f ) return 1;
//    std::cout<< "Testing operator -"<<std::endl;
//    vec2 = vec2 - vec;
//    if ( vec2[0] != 0.6f ) return 1;
//    if ( vec2[1] != 1.5f ) return 1;
//    if ( vec2[2] != 2.4f ) return 1;
    std::cout<< "Testing operator *"<<std::endl;
    Vec3d vec3 = vec2 * 5;
    if ( vec3[0] != 30.f ) return 1;
    if ( vec3[1] != 35.f ) return 1;
    if ( vec3[2] != 40.f ) return 1;
    std::cout<< "Testing operator /"<<std::endl;
    vec3 = vec3 / 5;
    if ( vec3[0] != 6.f) return 1;
    if ( vec3[1] != 7.f ) return 1;
    if ( vec3[2] != 8.f ) return 1;
    Vec3d b1 = { 1, 2, 3.3 };
    Vec3d b2 = { 1, 2, 3.3 };
    Vec3d b3 = { 1, 2.2, 3.3 };
    std::cout<< "Testing operator !="<<std::endl;
    if ( b1 != b2 ) return 1;
    std::cout<< "Testing operator =="<<std::endl;
    if ( b1 == b3 ) return 1;
    return 0;
}

int main() {
    int res = 0;
    res += testGet();
    res += testSet();
    res += testOperators();
    return res;
}