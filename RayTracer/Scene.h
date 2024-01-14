#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include <vector>
#include "Shape.h"
#include "Light.h"

class Scene {
public:
//    std::vector<Shape*>::const_iterator begin() const;
//    std::vector<Shape*>::const_iterator end() const;
//    std::vector<Light*>::const_iterator begin() const;
//    std::vector<Light*>::const_iterator end() const;
    std::vector<Shape*> shapes;
    std::vector<Light*> lights;
public:
};

#endif //COLLECTION_SCENE_H
