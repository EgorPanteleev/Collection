#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include <vector>
#include "BaseMesh.h"
#include "Light.h"
#include "Triangle.h"

class Scene {
public:
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
    void fillTriangles();
    std::vector<Triangle> getTriangles() const;
public:
    std::vector<Triangle> triangles;
};

#endif //COLLECTION_SCENE_H
