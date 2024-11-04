//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_MESH_H
#define COLLECTION_MESH_H
#include <iostream>
#include "Ray.h"
#include "Primitive.h"
#include "Triangle.h"
#include "Sphere.h"
#include <limits>
#include "Vector.h"
#include "IntersectionData.h"
#include "Material.h"
#include "BBox.h"
class Mesh {
public:
    Mesh();
    void loadMesh( const std::string& path );
    void rotate( const Vec3d& axis, double angle, bool group = false );
    void move( const Vec3d& p );
    void moveTo( const Vec3d& point );
    void scale( double scaleValue, bool group = false );
    void scale( const Vec3d& scaleVec, bool group = false );
    void scaleTo( double scaleValue, bool group = false );
    void scaleTo( const Vec3d& scaleVec, bool group = false);

    void setMinPoint( const Vec3d& vec, int ind = -1 );

    void setMaxPoint( const Vec3d& vec, int ind = -1 );

    [[nodiscard]] Vector <Primitive*> getPrimitives();
    [[nodiscard]] Vec3d getSamplePoint();
    [[nodiscard]] BBox getBBox() const;
    [[nodiscard]] Vec3d getOrigin() const;
    [[nodiscard]] bool isContainPoint( const Vec3d& p ) const;
//    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const;
    void setPrimitives( Vector<Primitive*>& _triangles );
    void addPrimitive( Primitive* primitive );
    void setMaterial( const Material& _material );
    [[nodiscard]] Material getMaterial() const;
protected:
    Vector<Primitive*> primitives;
    Material material;
};

#endif //COLLECTION_MESH_H
