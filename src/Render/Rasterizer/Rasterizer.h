//
// Created by auser on 7/29/24.
//

#ifndef COLLECTION_RASTERIZER_H
#define COLLECTION_RASTERIZER_H
#include "Vec3.h"
#include "Vec2.h"
#include "RGB.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
#include <Kokkos_Core.hpp>

class Rasterizer {
public:
    Rasterizer( Camera* c, Scene* s, Canvas* _canvas );

    ~Rasterizer();

    void drawLine( Vec2d v1, Vec2d v2, RGB color);

    void drawFilledTriangle( Triangle tri, RGB color);

    void render();

    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
private:

    Vec3d transform( Vec3d p );

    void clear();

    double** zBuffer;
    Kokkos::View<Camera*> camera;
    Kokkos::View<Scene*> scene;
    Kokkos::View<Canvas*> canvas;
};


struct RenderFunctor1 {

    RenderFunctor1( Rasterizer* _rasterizer );


    KOKKOS_INLINE_FUNCTION void operator()(const int i) const;

    Rasterizer* rasterizer;

    Mesh* mesh;
};


#endif //COLLECTION_RASTERIZER_H
