//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_MESH_H
#define COLLECTION_MESH_H
#include <iostream>
#include "Ray.h"
//#include "Primitive.h"
#include "Triangle.h"
#include <limits>
#include "Vector.h"
#include "IntersectionData.h"
#include "Material.h"
#include "BBox.h"
class Mesh {
public:
    Mesh();
    void loadMesh( const std::string& path );
    void rotate( const Vector3f& axis, float angle );
    void move( const Vector3f& p );
    void moveTo( const Vector3f& point );
    void scale( float scaleValue );
    void scale( const Vector3f& scaleVec );
    void scaleTo( float scaleValue );
    void scaleTo( const Vector3f& scaleVec );

    void setMinPoint( const Vector3f& vec, int ind = -1 );

    void setMaxPoint( const Vector3f& vec, int ind = -1 );

    [[nodiscard]] Vector <Triangle> getTriangles();
    [[nodiscard]] Vector3f getSamplePoint();
    [[nodiscard]] BBox getBBox() const;
    [[nodiscard]] Vector3f getOrigin() const;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const;
    void setTriangles( Vector<Triangle>& _triangles );
    void addTriangle( const Triangle& triangle );
    void setMaterial( const Material& _material );
    [[nodiscard]] Material getMaterial() const;
protected:
    Vector<Triangle> triangles;
    Material material;
};

#endif //COLLECTION_MESH_H
