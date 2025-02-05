#include "OBJLoader.h"
#include "OBJ_Loader.h"

bool OBJLoader::load( const std::string& path, HittableList* world, Material* material ) {
    size_t oldSize = world->hittables.size();
    static Lambertian defaultMaterial( { 0.8, 0.8, 0.8 } );
    if ( material == nullptr ) material = &defaultMaterial;
    objl::Loader loader;
    bool status = loader.LoadFile( path);
    if ( !status ) return status;
    for ( auto i = 0; i < loader.LoadedMeshes.size(); ++i ) {
        objl::Mesh curMesh = loader.LoadedMeshes[i];
        for ( auto j = 0; j < curMesh.Indices.size(); j += 3 ) {
            if ( j + 2 >= curMesh.Indices.size() ) continue;
            Point3d vertices[3];
            for ( int n = 0; n < 3; ++n ) {
                uint idx = curMesh.Indices[j + n];
                vertices[n] = { curMesh.Vertices[idx].Position.X, curMesh.Vertices[idx].Position.Y, curMesh.Vertices[idx].Position.Z };
            }

            world->add( new Triangle(vertices[0], vertices[1], vertices[2], material ) );
        }
    }
    std::cout << "Model Loaded with " << world->hittables.size() - oldSize << " hittables." << std::endl;
    return status;
}
