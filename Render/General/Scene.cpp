#include "Scene.h"
#include "Triangle.h"

__host__ __device__ void Scene::fillTriangles() {
    for ( auto mesh: meshes ) {
        for ( auto& triangle : mesh->getTriangles() ) triangles.push_back( triangle );
    }
}

__host__ __device__ Vector<BaseMesh*> Scene::getMeshes() const {
    return meshes;
}

__host__ __device__ Vector<Triangle> Scene::getTriangles() const {
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