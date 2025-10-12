//
// Created by auser on 8/28/24.
//

#ifndef COLLECTION_PRIMITIVE_H
#define COLLECTION_PRIMITIVE_H
#include "Vec3.h"
#include "BBox.h"
#include "Material.h"
#include "Ray.h"

class Primitive {
public:
    [[nodiscard]] BBox getBBox() const;
    [[nodiscard]] Vec3d getOrigin() const;
    [[nodiscard]] RGB getColor( const Vec3d& P ) const;
    [[nodiscard]] RGB getAmbient( const Vec3d& P ) const;
    [[nodiscard]] double getRoughness( const Vec3d& P ) const;
    [[nodiscard]] double getMetalness( const Vec3d& P ) const;
    [[nodiscard]] Material getMaterial() const;
    void setMaterial( const Material& mat );
    virtual void rotate( const Vec3d& axis, double angle ) = 0;
    virtual void move( const Vec3d& p ) = 0;
    virtual void moveTo( const Vec3d& point ) = 0;
    virtual void scale( double scaleValue ) = 0;
    virtual void scale( const Vec3d& scaleVec ) = 0;
    virtual void scaleTo( double scaleValue ) = 0;
    virtual void scaleTo( const Vec3d& scaleVec ) = 0;
    [[nodiscard]] virtual Vec3d getSamplePoint() const = 0;
    [[nodiscard]] virtual bool isContainPoint( const Vec3d& p ) const = 0;
    [[nodiscard]] virtual double intersectsWithRay( const Ray& ray ) const = 0;
    [[nodiscard]] virtual Vec3d getNormal( const Vec3d& P ) const = 0;
    [[nodiscard]] virtual Vec3d getV1() const { return {}; }
    [[nodiscard]] virtual Vec3d getV2() const { return {}; }
    [[nodiscard]] virtual Vec3d getV3() const { return {}; }
protected:
    [[nodiscard]] virtual int getIndex( const Vec3d& P, const ImageData& imageData ) const = 0;
    Material material;
    Vec3d origin;
    BBox bbox;
};


#endif //COLLECTION_PRIMITIVE_H
