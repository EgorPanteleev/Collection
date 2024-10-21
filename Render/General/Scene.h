#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include <utility>
#include "Vector.h"
#include "Mesh.h"
#include "Light.h"
#include "Primitive.h"
#include "Triangle.h"
#include "Sphere.h"

class LightInstance {
public:
    enum Type {
        COMMON_LIGHT,
        MESH_LIGHT
    };
    ~LightInstance() {
        light = nullptr;
        meshLight = nullptr;
    }
    LightInstance( Light* _light ): light( _light ), type( COMMON_LIGHT ), meshLight( nullptr ) {}
    LightInstance(Mesh* _light ): meshLight(_light ), type(MESH_LIGHT ), light(nullptr ) {}
    Type getType() { return type; }
    double getIntensity() {
        switch (type) {
            case COMMON_LIGHT: return light->intensity;
            case MESH_LIGHT: return meshLight->getMaterial().getIntensity();
        }
    }
    Vec3d getSamplePoint() {
        switch (type) {
            case COMMON_LIGHT: return light->getSamplePoint();
            case MESH_LIGHT: return meshLight->getSamplePoint();
        }
    }
    RGB getColor() {
        switch (type) {
            case COMMON_LIGHT: return light->lightColor;
            case MESH_LIGHT: return meshLight->getMaterial().getColor();
        }
    }
private:
    Type type;
    Light* light;
    Mesh* meshLight;
};


class Scene {
public:
    Scene(Vector<Mesh*> meshes, Vector<Sphere> spheres, Vector<Light*> lights );
    Scene();
    ~Scene();
    Sphere* add( Sphere* sph );
    Mesh* add(Mesh* mesh );
    Light* add( Light* light );
    [[nodiscard]] Vector<Mesh*> getMeshes() const;
    [[nodiscard]] Vector<Primitive*> getPrimitives() const;
    [[nodiscard]] Vector<LightInstance*> getLights() const;

private:
    Vector<Mesh*> meshes;
    Vector<LightInstance*> lights;
    Vector<Primitive*> primitives;
};

#endif //COLLECTION_SCENE_H
