#include "OBJLoader.h"
#include "OBJ_Loader.h"
#include <random>
OBJLoader::OBJLoader( const std::string& path, TriangularMesh* target ) {
    load( path, target );
}

bool OBJLoader::load( const std::string& path, TriangularMesh* target ) {
    objl::Loader loader;
    bool res = loader.LoadFile( path);
    if ( !res ) return res;
    for (int i = 0; i < loader.LoadedMeshes.size(); i++) {
        // Copy one of the loaded meshes to be our current mesh
        objl::Mesh curMesh = loader.LoadedMeshes[i];
        srand (static_cast <unsigned> (time(nullptr)));
        for (int j = 0; j < curMesh.Vertices.size(); j+=3 ) {
            Triangle triangle( { Vector3f(curMesh.Vertices[j].Position.X * 1000000, curMesh.Vertices[j].Position.Y * 1000000, curMesh.Vertices[j].Position.Z * 1000000),
                                 Vector3f(curMesh.Vertices[j+1].Position.X * 1000000, curMesh.Vertices[j+1].Position.Y * 1000000, curMesh.Vertices[j+1].Position.Z * 1000000),
                                 Vector3f(curMesh.Vertices[j+2].Position.X * 1000000, curMesh.Vertices[j+2].Position.Y * 1000000, curMesh.Vertices[j+2].Position.Z * 1000000) });

            static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            triangle.material = { RGB( static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 255,
                                       static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 255,
                                       static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 125 ), 1 , 0 };
            target->addTriangle( triangle);
        }
    }
    target->moveTo( Vector3f( 0,0,0) );
    target->scaleTo( 300 );
    std::cout << "Model Loaded."<<std::endl;
    return res;
}