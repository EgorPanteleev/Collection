//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_BASEMESH_H
#define COLLECTION_BASEMESH_H
#include <iostream>
#include "Ray.h"
#include <limits>
#include "Triangle.h"
#include "Vector.h"
#include "IntersectionData.h"
#include "Material.h"
#include "BBoxData.h"
class Triangle;
class IntersectionData;
class BaseMesh {
public:
    BaseMesh();
    virtual void loadMesh( const std::string& path );
    virtual void rotate( const Vector3f& axis, float angle ) = 0;
    virtual void move( const Vector3f& p ) = 0;
    virtual void moveTo( const Vector3f& point ) = 0;
    virtual void scale( float scaleValue ) = 0;
    virtual void scale( const Vector3f& scaleVec ) = 0;
    virtual void scaleTo( float scaleValue ) = 0;
    virtual void scaleTo( const Vector3f& scaleVec ) = 0;

    virtual void setMinPoint( const Vector3f& vec, int ind = -1 ) {  };

    virtual void setMaxPoint( const Vector3f& vec, int ind = -1 ) {  };

    virtual Vector <Triangle> getTriangles();
    [[nodiscard]] virtual BBoxData getBBox() const = 0;
    [[nodiscard]] virtual Vector3f getOrigin() const = 0;
    [[nodiscard]] virtual bool isContainPoint( const Vector3f& p ) const = 0;
    [[nodiscard]] virtual IntersectionData intersectsWithRay( const Ray& ray ) const = 0;
    [[nodiscard]] virtual Vector3f getNormal( const Vector3f& p ) const = 0;
    void setMaterial( const Material& _material );
    [[nodiscard]] Material getMaterial() const;
protected:
    Material material;
};

#endif //COLLECTION_BASEMESH_H
