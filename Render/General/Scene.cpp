#include "Scene.h"
#include "Triangle.h"

Scene::Scene( Vector<BaseMesh*> _meshes, Vector<Sphere> _spheres, Vector<Light*> _lights ) {
    meshes = _meshes;
    fillTriangles();
    spheres = _spheres;
    lights = _lights;
    for ( auto mesh: meshes )
        if ( mesh->getMaterial().getIntensity() != 0 ) lightMeshes.push_back( mesh );
    for ( const auto& sphere: spheres )
        if ( sphere.material.getIntensity() != 0 ) lightSpheres.push_back( sphere );
}

Scene::Scene(): meshes(), spheres(), lights(), lightSpheres(), lightMeshes(), triangles() {

}
void Scene::add( const Sphere& sphere ) {
    spheres.push_back( sphere );
    if ( sphere.material.getIntensity() != 0 )
        lightSpheres.push_back( sphere );
}
void Scene::add( BaseMesh* mesh ) {
    meshes.push_back( mesh );
    for ( auto& triangle: mesh->getTriangles() ) triangles.push_back( triangle );
    if ( mesh->getMaterial().getIntensity() != 0 )
        lightMeshes.push_back( mesh );
}
void Scene::add( Light* light ) {
    lights.push_back( light );
}

void Scene::fillTriangles() {
    for ( auto mesh: meshes ) {
        for ( auto& triangle : mesh->getTriangles() ) triangles.push_back( triangle );
    }
}

[[nodiscard]] Vector<Sphere> Scene::getSpheres() const { return spheres; }

Vector<BaseMesh*> Scene::getMeshes() const {
    return meshes;
}

Vector<Triangle> Scene::getTriangles() const {
    return triangles;
}

Vector<Light*> Scene::getLights() const {
    return lights;
}

Vector<BaseMesh*> Scene::getLightMeshes() const {
    return lightMeshes;
}
Vector<Sphere> Scene::getLightSpheres() const {
    return lightSpheres;
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