//
// Created by auser on 8/26/24.
//

#ifndef COLLECTION_GROUPOFMESHES_H
#define COLLECTION_GROUPOFMESHES_H
#include "Mesh.h"

class GroupOfMeshes {
public:
    GroupOfMeshes();
    void loadMesh( const std::string& path );
    void rotate( const Vec3d& axis, double angle );
    void move( const Vec3d& p );
    void moveTo( const Vec3d& point );
    void scale( double scaleValue );
    void scale( const Vec3d& scaleVec );
    void scaleTo( double scaleValue );
    void scaleTo( const Vec3d& scaleVec );

    void setMinPoint( const Vec3d& vec, int ind = -1 );
    void setMaxPoint( const Vec3d& vec, int ind = -1 );

    [[nodiscard]] BBox getBBox() const;
    [[nodiscard]] Vec3d getOrigin() const;

    Vector<Mesh*> getMeshes() const;
    void setMeshes( Vector<Mesh*>& _meshes );
    void addMesh( Mesh* mesh );
    void setMaterial( Material material, int index );
    void setMaterial( Material material );
protected:
    Vector<Mesh*> meshes;
};


#endif //COLLECTION_GROUPOFMESHES_H
