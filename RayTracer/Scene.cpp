#include "Scene.h"
#include "Triangle.h"

void Scene::fillTriangles() {
    for ( auto object: objects ) {
        for ( auto triangle : object->getTriangles() ) triangles.push_back( triangle );
    }
}

std::vector<Triangle> Scene::getTriangles() {
    return triangles;
}

//std::vector<Shape*>::const_iterator Scene::begin() const {
//    return shapes.begin();
//}
//std::vector<Shape*>::const_iterator Scene::end() const {
//    return shapes.end();
//}
//
//std::vector<Light*>::const_iterator Scene::begin() const {
//    return lights.begin();
//}
//std::vector<Light*>::const_iterator Scene::end() const {
//    return lights.end();
//}