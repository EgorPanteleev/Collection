#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
class RayTracer {
public:
    RayTracer( Camera* c, Scene* s );
    ~RayTracer();
    double computeLight( Vector3f P, Vector3f N );
    RGB traceRay( Ray& ray );
    void traceAllRays();
    Canvas* getCanvas() const;
private:
    Camera* cam;
    Scene* scene;
    Canvas* canvas;
};


#endif //COLLECTION_RAYTRACER_H
