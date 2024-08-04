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
    Scene( Vector<BaseMesh*> meshes, Vector<Sphere> spheres, Vector<Light*> lights );
    Scene();
    void add( const Sphere& sphere );
    void add( BaseMesh* mesh );
    void add( Light* light );
    [[nodiscard]] Vector<Sphere> getSpheres() const;
    [[nodiscard]] Vector<BaseMesh*> getMeshes() const;
    [[nodiscard]] Vector<Triangle> getTriangles() const;
    [[nodiscard]] Vector<Light*> getLights() const;
    [[nodiscard]] Vector<BaseMesh*> getLightMeshes() const;
    [[nodiscard]] Vector<Sphere> getLightSpheres() const;

private:
    void fillTriangles();
    Vector<BaseMesh*> meshes;
    Vector<Sphere> spheres;
    Vector<Light*> lights;
    Vector<BaseMesh*> lightMeshes;
    Vector<Sphere> lightSpheres;
    Vector<Triangle> triangles;
};

#endif //COLLECTION_SCENE_H
