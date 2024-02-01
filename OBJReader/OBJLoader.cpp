#include "OBJLoader.h"
#include "OBJ_Loader.h"
OBJLoader::OBJLoader( const std::string& path, OBJShape* target ) {
    load( path, target );
}

bool OBJLoader::load( const std::string& path, OBJShape* target ) {
    objl::Loader loader;
    bool res = loader.LoadFile( "C:/Users/igor/CLionProjects/Collection/OBJReader/model.obj");
    if ( !res ) return res;
    for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
        // Copy one of the loaded meshes to be our current mesh
        objl::Mesh curMesh = loader.LoadedMeshes[i];
        for (int j = 0; j < curMesh.Vertices.size(); j+=3 ) {
            target->triangles.emplace_back( Vector3f(curMesh.Vertices[j].Position.X * 1000000, curMesh.Vertices[j].Position.Y * 1000000, curMesh.Vertices[j].Position.Z * 1000000),
                                            Vector3f(curMesh.Vertices[j+1].Position.X * 1000000, curMesh.Vertices[j+1].Position.Y * 1000000, curMesh.Vertices[j+1].Position.Z * 1000000),
                                            Vector3f(curMesh.Vertices[j+2].Position.X * 1000000, curMesh.Vertices[j+2].Position.Y * 1000000, curMesh.Vertices[j+2].Position.Z * 1000000));
        }
    }
    return res;
}