//
// Created by auser on 7/29/24.
//

#ifndef COLLECTION_RASTERIZER_H
#define COLLECTION_RASTERIZER_H
#include "Vector3f.h"
#include "Vector2f.h"
#include "Color.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
#include <Kokkos_Core.hpp>

class Rasterizer {
public:
    Rasterizer( Camera* c, Scene* s, Canvas* _canvas );

    ~Rasterizer();

    void drawLine( Vector2f v1, Vector2f v2, RGB color);

    void drawFilledTriangle( Triangle tri, RGB color);

    void render();

    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
private:

    Vector3f transform( Vector3f p );

    void clear();

    float** zBuffer;
    Kokkos::View<Camera*> camera;
    Kokkos::View<Scene*> scene;
    Kokkos::View<Canvas*> canvas;
};


struct RenderFunctor1 {

    RenderFunctor1( Rasterizer* _rasterizer, BaseMesh* _mesh );


    KOKKOS_INLINE_FUNCTION void operator()(const int i) const;

    Rasterizer* rasterizer;

    BaseMesh* mesh;
};

struct RenderFunctor2 {

    RenderFunctor2( Rasterizer* _rasterizer );


    KOKKOS_INLINE_FUNCTION void operator()(const int i) const;

    Rasterizer* rasterizer;

};

#endif //COLLECTION_RASTERIZER_H
