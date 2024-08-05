#include "Scene.h"
#include "Triangle.h"

Scene::Scene(Vector<Mesh*> _meshes, Vector<Sphere> _spheres, Vector<Light*> _lights ) {
    for ( auto mesh: meshes ) add( mesh );
    for ( auto& sphere: spheres ) add( sphere );
    for ( auto light: _lights ) add( light) ;
}

Scene::Scene(): meshes(), spheres(), lights(), triangles() {

}

Scene::~Scene() {
//    for ( auto light: lights ) {
//        delete light;
//    }
}
Sphere Scene::add( Sphere sphere ) {
    spheres.push_back( sphere );
    if ( sphere.material.getIntensity() != 0 )
        lights.push_back( new LightInstance( &sphere ) );
    return sphere;
}

Mesh* Scene::add(Mesh* mesh ) {
    meshes.push_back( mesh );
    for ( auto& triangle: mesh->getTriangles() ) triangles.push_back( triangle );
    if ( mesh->getMaterial().getIntensity() != 0 )
        lights.push_back( new LightInstance( mesh ) );
    return mesh;
}
Light* Scene::add( Light* light ) {
    lights.push_back( new LightInstance( light ) );
    return light;
}

void Scene::fillTriangles() {
    for ( auto mesh: meshes ) {
        for ( auto& triangle : mesh->getTriangles() ) triangles.push_back( triangle );
    }
}

[[nodiscard]] Vector<Sphere> Scene::getSpheres() const { return spheres; }

Vector<Mesh*> Scene::getMeshes() const {
    return meshes;
}

Vector<Triangle> Scene::getTriangles() const {
    return triangles;
}

Vector<LightInstance*> Scene::getLights() const {
    return lights;
}


//Vector<Shape*>::const_iterator Scene::begin() const {
//    return shapes.begin();
//}
//Vector<Shape*>::const_iterator Scene::end() const {
//    return shapes.end();
//}
//
//Vector<Light*>::const_iterator Scene::begin() const {
//    return lights.begin();
//}
//Vector<Light*>::const_iterator Scene::end() const {
//    return lights.end();
//}