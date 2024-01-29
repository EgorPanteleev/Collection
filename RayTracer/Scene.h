#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include <vector>
#include "Object.h"
#include "Light.h"

class Scene {
public:
    std::vector<Object*> objects;
    std::vector<Light*> lights;
public:
};

#endif //COLLECTION_SCENE_H
