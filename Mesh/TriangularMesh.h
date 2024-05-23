//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_TRIANGULARMESH_H
#define COLLECTION_TRIANGULARMESH_H

#include "Vector3f.h"
#include "Ray.h"
#include "BaseMesh.h"


class BBoxData;
class TriangularMesh: public BaseMesh {
public:
//    TriangularMesh( std::vector<Triangle> _triangles );
//    TriangularMesh( const std::string& path );
    void loadMesh( const std::string& path ) override;
    void rotate( const Vector3f& axis, float angle ) override;
    void move( const Vector3f& p ) override;
    void moveTo( const Vector3f& point ) override;
    void scale( float scaleValue ) override;
    void scale( const Vector3f& scaleVec ) override;
    void scaleTo( float scaleValue ) override;
    void scaleTo( const Vector3f& scaleVec ) override;

    void setMinPoint( const Vector3f& vec, int ind = -1 ) override;

    void setMaxPoint( const Vector3f& vec, int ind = -1 ) override;

    std::vector<Triangle> getTriangles() override;
    [[nodiscard]] BBoxData getBBox() const override;
    [[nodiscard]] Vector3f getOrigin() const override;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
    void setTriangles( std::vector<Triangle>& _triangles );
    void addTriangle( const Triangle& triangle );
protected:
    std::vector<Triangle> triangles;
};

#endif //COLLECTION_TRIANGULARMESH_H