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
        for (int j = 0; j < curMesh.Vertices.size(); j+=3 ) {

            target->addTriangle( { Vector3f(curMesh.Vertices[j].Position.X , curMesh.Vertices[j].Position.Y , curMesh.Vertices[j].Position.Z ),
                                   Vector3f(curMesh.Vertices[j+1].Position.X , curMesh.Vertices[j+1].Position.Y , curMesh.Vertices[j+1].Position.Z ),
                                   Vector3f(curMesh.Vertices[j+2].Position.X , curMesh.Vertices[j+2].Position.Y , curMesh.Vertices[j+2].Position.Z ) });
        }
    }
    std::cout << target->getTriangles().size()<<std::endl;
    target->moveTo( Vector3f( 0,0,0) );
    auto bbox = target->getBBox();
    std::cout<< "Bbox data:" << std::endl << " min - ( " <<bbox.pMin.x << ", " << bbox.pMin.y << ", "<<bbox.pMin.z<< " )" <<std::endl
             << " max - ( " <<bbox.pMax.x << ", " << bbox.pMax.y << ", "<<bbox.pMax.z<< " )"<<std::endl;
    target->scaleTo( 300 );
    bbox = target->getBBox();
    std::cout << "Model Loaded."<<std::endl;
    std::cout<< "Bbox data:" << std::endl << " min - ( " <<bbox.pMin.x << ", " << bbox.pMin.y << ", "<<bbox.pMin.z<< " )" <<std::endl
    << " max - ( " <<bbox.pMax.x << ", " << bbox.pMax.y << ", "<<bbox.pMax.z<< " )"<<std::endl;
    return res;
}