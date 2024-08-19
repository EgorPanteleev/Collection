//
// Created by auser on 7/29/24.
//

#include "Rasterizer.h"

Rasterizer::Rasterizer( Camera* c, Scene* s, Canvas* _canvas ) {
    zBuffer = new float*[_canvas->getW()];
    for ( int i = 0; i < _canvas->getW(); i++ ) {
        zBuffer[i] = new float[_canvas->getH()];
    }
    for ( int i = 0; i < _canvas->getW(); i++ ) {
        for ( int j = 0; j < _canvas->getH(); j++ ) {
            zBuffer[i][j] = __FLT_MAX__;
        }
    }

    camera = Kokkos::View<Camera*>("camera");
    Kokkos::deep_copy(camera, *c);
    scene = Kokkos::View<Scene*>("scene");
    Kokkos::deep_copy(scene, *s);
    canvas = Kokkos::View<Canvas*>("canvas");
    Kokkos::deep_copy(canvas, *_canvas);
}

Rasterizer::~Rasterizer() {
}


void Rasterizer::drawLine( Vector2f v1, Vector2f v2, RGB color) {
    float k;
    if ( v2.getX() == v1.getX() ) k = __FLT_MAX__;
    else if ( v2.getY() == v1.getY() ) k = -__FLT_MAX__;
    else k = abs( v2.getY() - v1.getY() ) / abs( v2.getX() - v1.getX() );

    if ( k <= 1 ) {
        if ( k == -__FLT_MAX__ ) k = 0;
        if ( v1.getX() - v2.getX() > 0 ) std::swap( v1, v2 );
        int cf = v1.getY() - v2.getY() < 0 ? 1 : -1;
        for ( int x = (int) v1.getX(); x <= (int) v2.getX(); x++ ) {
            canvas(0).setColor( x, v1.getY() + k * ( x - v1.getX() ) * cf, color );
        }
    } else {
        if ( k == __FLT_MAX__ ) k = 0;
        else k = 1 / k;
        if ( v1.getY() - v2.getY() > 0 ) std::swap( v1, v2 );
        int cf = v1.getX() - v2.getX() < 0 ? 1 : -1;
        for ( int y = (int) v1.getY(); y <= (int) v2.getY(); y++ ) {
            canvas(0).setColor( v1.getX() + k * ( y - v1.getY() ) * cf, y, color );
        }
    }
}

Vector3f Rasterizer::transform( Vector3f p ) {
    p = p - camera(0).origin;
    float del = p.z == 0 ? 1 : p.z;
    return { p.x * (float) camera(0).dV / del + camera(0).Vx / 2,
             p.y * (float) camera(0).dV / del + camera(0).Vy/2,
             p.z                                                 };
}

void Rasterizer::clear() {
    for ( int i = 0; i < canvas(0).getW(); i++ ) {
        for ( int j = 0; j < canvas(0).getH(); j++ ) {
            zBuffer[i][j] = __FLT_MAX__;
            //canvas(0).setPixel( i, j, { 0, 0, 0 } );
        }
    }
}

float getZ( int x, int y, Vector3f v1, Vector3f v2, Vector3f v3 ) {
    float denominator = (v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y);
    float w1 = ((v2.y - v3.y) * (x - v3.x) + (v3.x - v2.x) * (y - v3.y)) / denominator;
    float w2 = ((v3.y - v1.y) * (x - v3.x) + (v1.x - v3.x) * (y - v3.y)) / denominator;
    float w3 = 1.0f - w1 - w2;
    return w1 * v1.z + w2 * v2.z + w3 * v3.z;
}

void Rasterizer::drawFilledTriangle( Triangle tri, RGB color) {
    Vector3f v1 = transform( tri.v1 );
    Vector3f v2 = transform( tri.v2 );
    Vector3f v3 = transform( tri.v3 );
    if ( abs( v1.getY() - v3.getY() ) < abs( v2.getY() - v3.getY() ) ) std::swap( v1, v2 );
    if ( abs( v1.getY() - v2.getY() ) < abs( v1.getY() - v3.getY() ) ) std::swap( v2, v3 );
    float dy = abs( v1.getY() - v2.getY() );
    int xMin, xMax;
    xMin = std::min( std::min(v1.getX(), v2.getX() ), v3.getX() );
    xMax = std::max( std::max(v1.getX(), v2.getX() ), v3.getX() );
    float k12;
    if ( v2.getX() == v1.getX() ) k12 = __FLT_MAX__;
    else if ( v2.getY() == v1.getY() ) k12 = -__FLT_MAX__;
    else k12 = ( v2.getY() - v1.getY() ) / ( v2.getX() - v1.getX() );
    float b12 = v1.getY() - k12 * v1.getX();

    float k13;
    if ( v3.getX() == v1.getX() ) k13 = __FLT_MAX__;
    else if ( v3.getY() == v1.getY() ) k13 = -__FLT_MAX__;
    else k13 = ( v3.getY() - v1.getY() ) / ( v3.getX() - v1.getX() );
    float b13 = v1.getY() - k13 * v1.getX();

    float k23;
    if ( v3.getX() == v2.getX() ) k23 = __FLT_MAX__;
    else if ( v3.getY() == v2.getY() ) k23 = -__FLT_MAX__;
    else k23 = ( v3.getY() - v2.getY() ) / ( v3.getX() - v2.getX() );
    float b23 = v2.getY() - k23 * v2.getX();
    Vector3f v11 = v1;
    Vector3f v22 = v2;
    if ( v11.getY() > v22.getY() ) std::swap( v11, v22 );
    int asd = v22.getY() > canvas(0).getH() ? canvas(0).getH() : v22.getY();
    int asdd= v11.getY() < 0 ? 0 : v11.getY();
    for ( int y = asdd; y < asd; y++) {
        float x12 = __FLT_MAX__;
        if ( k12 == __FLT_MAX__ && y < std::max( v1.getY(), v2.getY() ) && y > std::min( v1.getY(), v2.getY() ) ) x12 = std::floor( v1.getX() );
        else x12 = std::floor( ( y - b12 ) / k12 );

        float x13 = __FLT_MAX__;
        if ( k13 == __FLT_MAX__ && y < std::max( v1.getY(), v3.getY() ) && y > std::min( v1.getY(), v3.getY() )  ) x13 = std::floor( v1.getX() );
        else x13 = std::floor( ( y - b13 ) / k13 );

        float x23 = __FLT_MAX__;
        if ( k23 == __FLT_MAX__ && y < std::max( v2.getY(), v3.getY() ) && y > std::min( v2.getY(), v3.getY() ) ) x23 = std::floor( v2.getX() );
        else x23 = std::floor( ( y - b23 ) / k23 );
        Vector<float> vals;
        if ( x12 <= xMax && x12 >= xMin ) vals.push_back( x12 );
        if ( x13 <= xMax && x13 >= xMin ) vals.push_back( x13 );
        if ( x23 <= xMax && x23 >= xMin ) vals.push_back( x23 );
        if ( vals.size() <= 1 ) continue;
        if ( vals[0] == vals[1] && vals.size() >= 3 ) std::swap( vals[1], vals[2] );
        if ( vals[0] > vals[1] ) std::swap( vals[0], vals[1] );
        int asd1 = vals[1] > canvas(0).getW() ? canvas(0).getW() : vals[1];
        int asdd1= vals[0] < 0 ? 0 : vals[0];
        for ( int x = asdd1; x < asd1; x++ ) {
            float z = getZ( x, y, v1, v2, v3 );
            if ( z <= 0 ) continue;
            if ( z <= zBuffer[x][y]) {
                zBuffer[x][y] = z;
                canvas(0).setColor( x, y, color );
            }
        }
    }

}

//struct BBox {
//    BBox( const Vector2f& _min, const Vector2f& _max ): min( _min ), max( _max ) {}
//    Vector2f min;
//    Vector2f max;
//};
//
//BBox getBBox( Triangle tri ) {
//    float minX = std::min( std::min( tri.v1[0], tri.v2[0] ), tri.v3[0] );
//    float maxX = std::max( std::max( tri.v1[0], tri.v2[0] ), tri.v3[0] );
//    float minY = std::min( std::min( tri.v1[1], tri.v2[1] ), tri.v3[1] );
//    float maxY = std::max( std::max( tri.v1[1], tri.v2[1] ), tri.v3[1] );
//    return { Vector2f( minX, minY ), Vector2f( maxX, maxY ) };
//}
//
//bool isPointInTriangle( Triangle tri, Vector2f p ) {
//    // Vectors from point A to point P
//    double v0x = tri.v3.x - tri.v1.x;
//    double v0y = tri.v3.y - tri.v1.y;
//    double v1x = tri.v2.x - tri.v1.x;
//    double v1y = tri.v2.y - tri.v1.y;
//    double v2x = p.getX() - tri.v1.x;
//    double v2y = p.getY() - tri.v1.y;
//
//    // Dot products
//    double dot00 = v0x * v0x + v0y * v0y;
//    double dot01 = v0x * v1x + v0y * v1y;
//    double dot02 = v0x * v2x + v0y * v2y;
//    double dot11 = v1x * v1x + v1y * v1y;
//    double dot12 = v1x * v2x + v1y * v2y;
//
//    // Barycentric coordinates
//    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
//    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
//    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
//
//    // Check if point is in triangle
//    return (u >= 0) && (v >= 0) && (u + v <= 1);
//}

//void Rasterizer::drawFilledTriangle( Triangle tri, RGB color) {
//    Vector3f v1 = transform( tri.v1 );
//    Vector3f v2 = transform( tri.v2 );
//    Vector3f v3 = transform( tri.v3 );
//    Triangle triangle( v1, v2, v3 );
//    BBox bbox = getBBox( triangle );
//    bbox.min = { bbox.min.getX() < 0 ? 0 : bbox.min.getX(), bbox.min.getY() < 0 ? 0 : bbox.min.getY() };
//    bbox.max = { bbox.max.getX() > canvas(0).getW() ? canvas(0).getW() : bbox.max.getX(),
//                 bbox.max.getY() > canvas(0).getH() ? canvas(0).getH() : bbox.max.getY() };
//    for ( int x = bbox.min.getX(); x < bbox.max.getX(); x++ ) {
//        for (int y = bbox.min.getY(); y < bbox.max.getY(); y++) {
//            if (!isPointInTriangle(triangle, Vector2f(x, y))) continue;
//            float z = getZ(x, y, v1, v2, v3);
//            if (z <= 0) continue;
//            if (z > zBuffer[x][y]) continue;
//            zBuffer[x][y] = z;
//            canvas(0).setPixel(x, y, color);
//        }
//    }
//}

//void drawHorizontalLine(int x1, int x2, int y, RGB color, Canvas* canvas ) {
//    if (x1 > x2) std::swap(x1, x2);
//    x1 = x1 < 0 ? 0 : x1;
//    x2 = x2 > canvas->getW() ? canvas->getW() : x2;
//    for (int x = x1; x < x2; ++x) {
//        canvas->setPixel( x, y, color );
//    }
//}
//
//void Rasterizer::drawFilledTriangle( Triangle tri, RGB color) {
//    Vector3f v0 = transform( tri.v1 );
//    Vector3f v1 = transform( tri.v2 );
//    Vector3f v2 = transform( tri.v3 );
//// Sort vertices by y-coordinate
//    if (v0.y > v1.y) std::swap(v0, v1);
//    if (v0.y > v2.y) std::swap(v0, v2);
//    if (v1.y > v2.y) std::swap(v1, v2);
//
//// Compute inverse slopes
//    float invslope1 = (v1.x - v0.x) / static_cast<float>(v1.y - v0.y);
//    float invslope2 = (v2.x - v0.x) / static_cast<float>(v2.y - v0.y);
//    float invslope3 = (v2.x - v1.x) / static_cast<float>(v2.y - v1.y);
//
//// Starting x-coordinates
//    float curx1 = v0.x;
//    float curx2 = v0.x;
//
//// Fill bottom part of the triangle (v0 to v1)
//    float asd1 = v0.y < 0 ? 0 : v0.y;
//    float asd2 = v1.y > canvas(0).getH() ? canvas(0).getH() : v1.y;
//    for (int y = asd1; y < asd2; ++y) {
//        drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), y, color, &canvas(0) );
//        curx1 += invslope1;
//        curx2 += invslope2;
//    }
//
//// Fill top part of the triangle (v1 to v2)
//    curx1 = v1.x;
//    asd1 = v1.y < 0 ? 0 : v1.y;
//    asd2 = v2.y > canvas(0).getH() ? canvas(0).getH() : v2.y;
//    for (int y = asd1; y < asd2; ++y) {
//        drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), y, color, &canvas(0) );
//        curx1 += invslope3;
//        curx2 += invslope2;
//    }
//}

//void Rasterizer::render() {
//    clear();
//    for ( auto mesh: scene(0).getMeshes() ) {
//        for ( const auto& triangle: mesh->getTriangles() ) {
//            drawFilledTriangle( triangle, mesh->getMaterial().getColor() );
//        }
//    }
//}


void Rasterizer::render() {
    RenderFunctor1 renderFunctor1( this );
    Kokkos::parallel_for("parallel1D", getScene()->getMeshes().size(), renderFunctor1);
}

RenderFunctor1::RenderFunctor1( Rasterizer* _rasterizer ): rasterizer( _rasterizer ) {
}

void RenderFunctor1::operator()(const int i ) const {
    auto mesh = rasterizer->getScene()->getMeshes()[i];
    for ( const auto& triangle:mesh->getTriangles() ) {
        rasterizer->drawFilledTriangle( triangle, mesh->getMaterial().getColor() );
    }
}

Canvas* Rasterizer::getCanvas() const {
    return &(canvas(0));
}

Scene* Rasterizer::getScene() const {
    return &(scene(0));
}
Camera* Rasterizer::getCamera() const {
    return &(camera(0));
}
