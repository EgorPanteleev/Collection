#include "Scene.h"
#include "Triangle.h"

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