#ifndef COLLECTION_SCENE_H
#define COLLECTION_SCENE_H
#include <iostream>
#include <utility>
#include "Vector.h"
#include "BaseMesh.h"
#include "Light.h"
#include "Triangle.h"
#include "Sphere.h"

class LightInstance {
public:
    enum Type {
        COMMON_LIGHT,
        MESH_LIGHT,
        SPHERE_LIGHT
    };
    ~LightInstance() {
        light = nullptr;
        meshLight = nullptr;
        sphereLight = {};
    }
    LightInstance( Light* _light ): light( _light ), type( COMMON_LIGHT ), meshLight( nullptr ), sphereLight() {}
    LightInstance( BaseMesh* _light ): meshLight( _light ), type( MESH_LIGHT ), light( nullptr ), sphereLight() {}
    LightInstance( Sphere* _light ): sphereLight( *_light ), type( SPHERE_LIGHT ), meshLight( nullptr ),light( nullptr ) {}
    Type getType() { return type; }
    float getIntensity() {
        switch (type) {
            case COMMON_LIGHT: return light->intensity;
            case MESH_LIGHT: return meshLight->getMaterial().getIntensity();
            case SPHERE_LIGHT: return sphereLight.getMaterial().getIntensity();
        }
    }
    Vector3f getSamplePoint() {
        switch (type) {
            case COMMON_LIGHT: return light->getSamplePoint();
            case MESH_LIGHT: return meshLight->getSamplePoint();
            case SPHERE_LIGHT: return sphereLight.getSamplePoint();
        }
    }
    RGB getColor() {
        switch (type) {
            case COMMON_LIGHT: return light->lightColor;
            case MESH_LIGHT: return meshLight->getMaterial().getColor();
            case SPHERE_LIGHT: return sphereLight.getMaterial().getColor();
        }
    }
private:
    Type type;
    Light* light;
    BaseMesh* meshLight;
    Sphere sphereLight;
};


class Scene {
public:
    Scene( Vector<BaseMesh*> meshes, Vector<Sphere> spheres, Vector<Light*> lights );
    Scene();
    ~Scene();
    void add( Sphere sphere );
    void add( BaseMesh* mesh );
    void add( Light* light );
    [[nodiscard]] Vector<Sphere> getSpheres() const;
    [[nodiscard]] Vector<BaseMesh*> getMeshes() const;
    [[nodiscard]] Vector<Triangle> getTriangles() const;
    [[nodiscard]] Vector<LightInstance*> getLights() const;

private:
    void fillTriangles();
    Vector<BaseMesh*> meshes;
    Vector<Sphere> spheres;
    Vector<LightInstance*> lights;
    Vector<Triangle> triangles;
};

#endif //COLLECTION_SCENE_H
