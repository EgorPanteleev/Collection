#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include <vector>
#include "Object.h"
#include "Light.h"
#include "Triangle.h"

class Scene {
public:
    std::vector<Object*> objects;
    std::vector<Light*> lights;
    void fillTriangles();
    std::vector<Triangle> getTriangles();
public:
    std::vector<Triangle> triangles;
};

#endif //COLLECTION_SCENE_H
