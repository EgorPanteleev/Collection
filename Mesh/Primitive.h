//
// Created by auser on 8/28/24.
//

#ifndef COLLECTION_PRIMITIVE_H
#define COLLECTION_PRIMITIVE_H
#include "Vector3f.h"
#include "BBox.h"
#include "Material.h"
#include "Ray.h"

class Primitive {
public:
    [[nodiscard]] BBox getBBox() const;
    [[nodiscard]] Vector3f getOrigin() const;
    [[nodiscard]] RGB getColor( const Vector3f& P ) const;
    [[nodiscard]] RGB getAmbient( const Vector3f& P ) const;
    [[nodiscard]] float getRoughness( const Vector3f& P ) const;
    [[nodiscard]] float getMetalness( const Vector3f& P ) const;
    [[nodiscard]] Material getMaterial() const;
    void setMaterial( const Material& mat );
    virtual void rotate( const Vector3f& axis, float angle ) = 0;
    virtual void move( const Vector3f& p ) = 0;
    virtual void moveTo( const Vector3f& point ) = 0;
    virtual void scale( float scaleValue ) = 0;
    virtual void scale( const Vector3f& scaleVec ) = 0;
    virtual void scaleTo( float scaleValue ) = 0;
    virtual void scaleTo( const Vector3f& scaleVec ) = 0;
    [[nodiscard]] virtual Vector3f getSamplePoint() const = 0;
    [[nodiscard]] virtual bool isContainPoint( const Vector3f& p ) const = 0;
    [[nodiscard]] virtual float intersectsWithRay( const Ray& ray ) const = 0;
    [[nodiscard]] virtual Vector3f getNormal( const Vector3f& P ) const = 0;
    [[nodiscard]] virtual Vector3f getV1() const { return {}; }
    [[nodiscard]] virtual Vector3f getV2() const { return {}; }
    [[nodiscard]] virtual Vector3f getV3() const { return {}; }
protected:
    [[nodiscard]] virtual int getIndex( const Vector3f& P, const ImageData& imageData ) const = 0;
    Material material;
    Vector3f origin;
    BBox bbox;
};


#endif //COLLECTION_PRIMITIVE_H
