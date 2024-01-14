#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
class RayTracer {
public:
    RayTracer( Scene* s );
    ~RayTracer();
    double computeLight( Point P, Point N );
    RGB traceRay( Ray& ray );
    void traceAllRays( Camera& cam );
    Canvas* getCanvas() const;
private:
    Scene* scene;
    Canvas* canvas;
};


#endif //COLLECTION_RAYTRACER_H
