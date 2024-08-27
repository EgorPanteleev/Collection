#include "OBJLoader.h"
#include "OBJ_Loader.h"
#include <random>
OBJLoader::OBJLoader( const std::string& path, Mesh* target ) {
    load( path, target );
}

OBJLoader::OBJLoader( const std::string& path, GroupOfMeshes* target ) {
    load( path, target );
}

bool OBJLoader::load( const std::string& path, Mesh* target ) {
    objl::Loader loader;
    bool res = loader.LoadFile( path);
    if ( !res ) return res;
    for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
        // Copy one of the loaded meshes to be our current mesh
        objl::Mesh curMesh = loader.LoadedMeshes[i];
        for (int j = 0; j < curMesh.Indices.size(); j += 3) {
            if ( j + 2 >= curMesh.Indices.size() ) continue;
            uint idx1 = (int) curMesh.Indices[j];
            uint idx2 = (int) curMesh.Indices[j + 1];
            uint idx3 = (int) curMesh.Indices[j + 2];
            target->addTriangle( { Vector3f(curMesh.Vertices[idx1].Position.X , curMesh.Vertices[idx1].Position.Y , curMesh.Vertices[idx1].Position.Z ),
                                   Vector3f(curMesh.Vertices[idx2].Position.X , curMesh.Vertices[idx2].Position.Y , curMesh.Vertices[idx2].Position.Z ),
                                   Vector3f(curMesh.Vertices[idx3].Position.X , curMesh.Vertices[idx3].Position.Y , curMesh.Vertices[idx3].Position.Z ) });
        }
    }
    std::cout << "Model Loaded with " << target->getTriangles().size() << " triangles." << std::endl;
    return res;
}

bool OBJLoader::load( const std::string& path, GroupOfMeshes* target ) {
    objl::Loader loader;
    bool res = loader.LoadFile( path);
    if ( !res ) return res;
    for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
        // Copy one of the loaded meshes to be our current mesh
        objl::Mesh curMesh = loader.LoadedMeshes[i];
        Mesh* newMesh = new Mesh();
        for (int j = 0; j < curMesh.Indices.size(); j += 3) {
            if ( j + 2 >= curMesh.Indices.size() ) continue;
            uint idx1 = (int) curMesh.Indices[j];
            uint idx2 = (int) curMesh.Indices[j + 1];
            uint idx3 = (int) curMesh.Indices[j + 2];
            newMesh->addTriangle( { Vector3f(curMesh.Vertices[idx1].Position.X , curMesh.Vertices[idx1].Position.Y , curMesh.Vertices[idx1].Position.Z ),
                                   Vector3f(curMesh.Vertices[idx2].Position.X , curMesh.Vertices[idx2].Position.Y , curMesh.Vertices[idx2].Position.Z ),
                                   Vector3f(curMesh.Vertices[idx3].Position.X , curMesh.Vertices[idx3].Position.Y , curMesh.Vertices[idx3].Position.Z ) });
        }
        target->addMesh( newMesh );
    }
    int triangleCount = 0;
    for ( auto mesh: target->getMeshes() ) {
        triangleCount += mesh->getTriangles().size();
    }
    std::cout << "Model Loaded with " << triangleCount << " triangles." << std::endl;
    return res;
}