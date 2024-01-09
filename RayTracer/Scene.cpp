#include "Scene.h"

std::vector<Shape*>::const_iterator Scene::begin() const {
    return shapes.begin();
}
std::vector<Shape*>::const_iterator Scene::end() const {
    return shapes.end();
}