//
// Created by auser on 8/28/24.
//

#include "Primitive.h"

BBox Primitive::getBBox() const {
    return bbox;
}
Vector3f Primitive::getOrigin() const {
    return origin;
}

RGB Primitive::getColor( const Vector3f& P ) const {
    if ( !material.getTexture().colorMap.data ) return material.getColor();

    int ind = getIndex( P, material.getTexture().colorMap );
    return {
            (float) material.getTexture().colorMap.data[ind    ] * 1.0f,
            (float) material.getTexture().colorMap.data[ind + 1] * 1.0f,
            (float) material.getTexture().colorMap.data[ind + 2] * 1.0f
    };
}
RGB Primitive::getAmbient( const Vector3f& P ) const {
    if ( !material.getTexture().ambientMap.data ) return { 1, 1, 1 };
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().ambientMap );
    return {
            (float) material.getTexture().ambientMap.data[ind    ] * F1_255,
            (float) material.getTexture().ambientMap.data[ind + 1] * F1_255,
            (float) material.getTexture().ambientMap.data[ind + 2] * F1_255
    };
}
float Primitive::getRoughness( const Vector3f& P ) const {
    if ( !material.getTexture().roughnessMap.data ) return material.getRoughness();
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().roughnessMap );

    return (float) material.getTexture().roughnessMap.data[ind] * F1_255;
}
float Primitive::getMetalness( const Vector3f& P ) const {
    if ( !material.getTexture().metalnessMap.data ) return material.getMetalness();
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().metalnessMap );

    return (float) material.getTexture().metalnessMap.data[ind] * F1_255;
}

Material Primitive::getMaterial() const {
    return material;
}
void Primitive::setMaterial( const Material& mat ) {
    material = mat;
}