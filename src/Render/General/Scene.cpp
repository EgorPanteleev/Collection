#include "Scene.h"
#include "Primitive.h"

Scene::Scene(Vector<Mesh*> _meshes, Vector<Sphere> _spheres, Vector<Light*> _lights ) {
    for ( auto mesh: meshes ) add( mesh );
    for ( auto light: _lights ) add( light) ;
}

Scene::Scene(): meshes(), lights(), primitives() {

}

Scene::~Scene() {
//    for ( auto light: lights ) {
//        delete light;
//    }
}

Sphere* Scene::add( Sphere* sph ) {
    primitives.push_back( sph );
    return sph;
}

Mesh* Scene::add(Mesh* mesh ) {
    meshes.push_back( mesh );
    for ( auto primitive: mesh->getPrimitives() ) primitives.push_back( primitive );
    if ( mesh->getMaterial().getIntensity() != 0 )
        lights.push_back( new LightInstance( mesh ) );
    return mesh;
}
Light* Scene::add( Light* light ) {
    lights.push_back( new LightInstance( light ) );
    return light;
}

Vector<Mesh*> Scene::getMeshes() const {
    return meshes;
}

Vector<Primitive*> Scene::getPrimitives() const {
    return primitives;
}

Vector<LightInstance*> Scene::getLights() const {
    return lights;
}
