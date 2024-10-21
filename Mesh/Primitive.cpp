//
// Created by auser on 8/28/24.
//

#include "Primitive.h"

BBox Primitive::getBBox() const {
    return bbox;
}
Vec3d Primitive::getOrigin() const {
    return origin;
}

RGB Primitive::getColor( const Vec3d& P ) const {
    if ( !material.getTexture().colorMap.data ) return material.getColor();

    int ind = getIndex( P, material.getTexture().colorMap );
    return {
            material.getTexture().colorMap.data[ind    ] * 1.0,
            material.getTexture().colorMap.data[ind + 1] * 1.0,
            material.getTexture().colorMap.data[ind + 2] * 1.0
    };
}
RGB Primitive::getAmbient( const Vec3d& P ) const {
    if ( !material.getTexture().ambientMap.data ) return { 1, 1, 1 };
    constexpr double F1_255 = 1 / 255.0;
    int ind = getIndex( P, material.getTexture().ambientMap );
    return {
            material.getTexture().ambientMap.data[ind    ] * F1_255,
            material.getTexture().ambientMap.data[ind + 1] * F1_255,
            material.getTexture().ambientMap.data[ind + 2] * F1_255
    };
}
double Primitive::getRoughness( const Vec3d& P ) const {
    if ( !material.getTexture().roughnessMap.data ) return material.getRoughness();
    constexpr double F1_255 = 1 / 255.0;
    int ind = getIndex( P, material.getTexture().roughnessMap );

    return material.getTexture().roughnessMap.data[ind] * F1_255;
}
double Primitive::getMetalness( const Vec3d& P ) const {
    if ( !material.getTexture().metalnessMap.data ) return material.getMetalness();
    constexpr double F1_255 = 1 / 255.0;
    int ind = getIndex( P, material.getTexture().metalnessMap );

    return material.getTexture().metalnessMap.data[ind] * F1_255;
}

Material Primitive::getMaterial() const {
    return material;
}
void Primitive::setMaterial( const Material& mat ) {
    material = mat;
}