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
    void rotate( const Vector3f& axis, float angle );
    void move( const Vector3f& p );
    void moveTo( const Vector3f& point );
    void scale( float scaleValue );
    void scale( const Vector3f& scaleVec );
    void scaleTo( float scaleValue );
    void scaleTo( const Vector3f& scaleVec );

    void setMinPoint( const Vector3f& vec, int ind = -1 );
    void setMaxPoint( const Vector3f& vec, int ind = -1 );

    [[nodiscard]] BBox getBBox() const;
    [[nodiscard]] Vector3f getOrigin() const;

    Vector<Mesh*> getMeshes() const;
    void setMeshes( Vector<Mesh*>& _meshes );
    void addMesh( Mesh* mesh );
    void setMaterial( Material material, int index );
    void setMaterial( Material material );
protected:
    Vector<Mesh*> meshes;
};


#endif //COLLECTION_GROUPOFMESHES_H
