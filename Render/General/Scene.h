#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include "Vector.h"
#include "BaseMesh.h"
#include "Light.h"
#include "Triangle.h"

class Scene {
public:
    __host__ __device__ Vector<BaseMesh*> meshes;
    __host__ __device__ Vector<Light*> lights;
    __host__ __device__ void fillTriangles();
    [[nodiscard]] __host__ __device__ Vector<BaseMesh*> getMeshes() const;
    [[nodiscard]] __host__ __device__ Vector<Triangle> getTriangles() const;
public:
    Vector<Triangle> triangles;
};

#endif //COLLECTION_SCENE_H
