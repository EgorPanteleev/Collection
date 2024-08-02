#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include "Vector.h"
#include "BaseMesh.h"
#include "Light.h"
#include "Triangle.h"
#include "Sphere.h"

class Scene {
public:
    Vector<BaseMesh*> meshes;
    Vector<Sphere> spheres;
    Vector<Light*> lights;
    void fillTriangles();
    [[nodiscard]] Vector<Sphere> getSpheres() const;
    [[nodiscard]] Vector<BaseMesh*> getMeshes() const;
    [[nodiscard]] Vector<Triangle> getTriangles() const;
public:
    Vector<Triangle> triangles;
};

#endif //COLLECTION_SCENE_H
