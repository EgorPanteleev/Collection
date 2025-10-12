//
// Created by auser on 7/29/24.
//

#include "Rasterizer.h"

Rasterizer::Rasterizer( Camera* c, Scene* s, Canvas* _canvas ) {
    zBuffer = new double*[_canvas->getW()];
    for ( int i = 0; i < _canvas->getW(); i++ ) {
        zBuffer[i] = new double[_canvas->getH()];
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


void Rasterizer::drawLine( Vec2d v1, Vec2d v2, RGB color) {
    double k;
    if ( v2[0] == v1[0] ) k = __FLT_MAX__;
    else if ( v2[1] == v1[1] ) k = -__FLT_MAX__;
    else k = abs( v2[1] - v1[1] ) / abs( v2[0] - v1[0] );

    if ( k <= 1 ) {
        if ( k == -__FLT_MAX__ ) k = 0;
        if ( v1[0] - v2[0] > 0 ) std::swap( v1, v2 );
        int cf = v1[1] - v2[1] < 0 ? 1 : -1;
        for ( int x = (int) v1[0]; x <= (int) v2[0]; x++ ) {
            canvas(0).setColor( x, v1[1] + k * ( x - v1[0] ) * cf, color );
        }
    } else {
        if ( k == __FLT_MAX__ ) k = 0;
        else k = 1 / k;
        if ( v1[1] - v2[1] > 0 ) std::swap( v1, v2 );
        int cf = v1[0] - v2[0] < 0 ? 1 : -1;
        for ( int y = (int) v1[1]; y <= (int) v2[1]; y++ ) {
            canvas(0).setColor( v1[0] + k * ( y - v1[1] ) * cf, y, color );
        }
    }
}

Vec3d Rasterizer::transform( Vec3d p ) {
    p = p - camera(0).origin;
    double del = p[2] == 0 ? 1 : p[2];
    return { p[0] * (double) camera(0).dV / del + camera(0).Vx / 2,
             p[1] * (double) camera(0).dV / del + camera(0).Vy/2,
             p[2]                                                 };
}

void Rasterizer::clear() {
    for ( int i = 0; i < canvas(0).getW(); i++ ) {
        for ( int j = 0; j < canvas(0).getH(); j++ ) {
            zBuffer[i][j] = __FLT_MAX__;
            //canvas(0).setPixel( i, j, { 0, 0, 0 } );
        }
    }
}

double getZ( int x, int y, Vec3d v1, Vec3d v2, Vec3d v3 ) {
    double denominator = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1]);
    double w1 = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / denominator;
    double w2 = ((v3[1] - v1[1]) * (x - v3[0]) + (v1[0] - v3[0]) * (y - v3[1])) / denominator;
    double w3 = 1.0f - w1 - w2;
    return w1 * v1[2] + w2 * v2[2] + w3 * v3[2];
}

void Rasterizer::drawFilledTriangle( Triangle tri, RGB color) {
    Vec3d v1 = transform( tri.v1 );
    Vec3d v2 = transform( tri.v2 );
    Vec3d v3 = transform( tri.v3 );
    if ( abs( v1[1] - v3[1] ) < abs( v2[1] - v3[1] ) ) std::swap( v1, v2 );
    if ( abs( v1[1] - v2[1] ) < abs( v1[1] - v3[1] ) ) std::swap( v2, v3 );
    double dy = abs( v1[1] - v2[1] );
    int xMin, xMax;
    xMin = std::min( std::min(v1[0], v2[0] ), v3[0] );
    xMax = std::max( std::max(v1[0], v2[0] ), v3[0] );
    double k12;
    if ( v2[0] == v1[0] ) k12 = __FLT_MAX__;
    else if ( v2[1] == v1[1] ) k12 = -__FLT_MAX__;
    else k12 = ( v2[1] - v1[1] ) / ( v2[0] - v1[0] );
    double b12 = v1[1] - k12 * v1[0];

    double k13;
    if ( v3[0] == v1[0] ) k13 = __FLT_MAX__;
    else if ( v3[1] == v1[1] ) k13 = -__FLT_MAX__;
    else k13 = ( v3[1] - v1[1] ) / ( v3[0] - v1[0] );
    double b13 = v1[1] - k13 * v1[0];

    double k23;
    if ( v3[0] == v2[0] ) k23 = __FLT_MAX__;
    else if ( v3[1] == v2[1] ) k23 = -__FLT_MAX__;
    else k23 = ( v3[1] - v2[1] ) / ( v3[0] - v2[0] );
    double b23 = v2[1] - k23 * v2[0];
    Vec3d v11 = v1;
    Vec3d v22 = v2;
    if ( v11[1] > v22[1] ) std::swap( v11, v22 );
    int asd = v22[1] > canvas(0).getH() ? canvas(0).getH() : v22[1];
    int asdd= v11[1] < 0 ? 0 : v11[1];
    for ( int y = asdd; y < asd; y++) {
        double x12 = __FLT_MAX__;
        if ( k12 == __FLT_MAX__ && y < std::max( v1[1], v2[1] ) && y > std::min( v1[1], v2[1] ) ) x12 = std::floor( v1[0] );
        else x12 = std::floor( ( y - b12 ) / k12 );

        double x13 = __FLT_MAX__;
        if ( k13 == __FLT_MAX__ && y < std::max( v1[1], v3[1] ) && y > std::min( v1[1], v3[1] )  ) x13 = std::floor( v1[0] );
        else x13 = std::floor( ( y - b13 ) / k13 );

        double x23 = __FLT_MAX__;
        if ( k23 == __FLT_MAX__ && y < std::max( v2[1], v3[1] ) && y > std::min( v2[1], v3[1] ) ) x23 = std::floor( v2[0] );
        else x23 = std::floor( ( y - b23 ) / k23 );
        Vector<double> vals;
        if ( x12 <= xMax && x12 >= xMin ) vals.push_back( x12 );
        if ( x13 <= xMax && x13 >= xMin ) vals.push_back( x13 );
        if ( x23 <= xMax && x23 >= xMin ) vals.push_back( x23 );
        if ( vals.size() <= 1 ) continue;
        if ( vals[0] == vals[1] && vals.size() >= 3 ) std::swap( vals[1], vals[2] );
        if ( vals[0] > vals[1] ) std::swap( vals[0], vals[1] );
        int asd1 = vals[1] > canvas(0).getW() ? canvas(0).getW() : vals[1];
        int asdd1= vals[0] < 0 ? 0 : vals[0];
        for ( int x = asdd1; x < asd1; x++ ) {
            double z = getZ( x, y, v1, v2, v3 );
            if ( z <= 0 ) continue;
            if ( z <= zBuffer[x][y]) {
                zBuffer[x][y] = z;
                canvas(0).setColor( x, y, color );
            }
        }
    }

}

//struct BBox {
//    BBox( const Vec2d& _min, const Vec2d& _max ): min( _min ), max( _max ) {}
//    Vec2d min;
//    Vec2d max;
//};
//
//BBox getBBox( Triangle tri ) {
//    double minX = std::min( std::min( tri.v1[0], tri.v2[0] ), tri.v3[0] );
//    double maxX = std::max( std::max( tri.v1[0], tri.v2[0] ), tri.v3[0] );
//    double minY = std::min( std::min( tri.v1[1], tri.v2[1] ), tri.v3[1] );
//    double maxY = std::max( std::max( tri.v1[1], tri.v2[1] ), tri.v3[1] );
//    return { Vec2d( minX, minY ), Vec2d( maxX, maxY ) };
//}
//
//bool isPointInTriangle( Triangle tri, Vec2d p ) {
//    // Vectors from point A to point P
//    double v0x = tri.v3[0] - tri.v1[0];
//    double v0y = tri.v3[1] - tri.v1[1];
//    double v1x = tri.v2[0] - tri.v1[0];
//    double v1y = tri.v2[1] - tri.v1[1];
//    double v2x = p[0] - tri.v1[0];
//    double v2y = p[1] - tri.v1[1];
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
//    Vec3d v1 = transform( tri.v1 );
//    Vec3d v2 = transform( tri.v2 );
//    Vec3d v3 = transform( tri.v3 );
//    Triangle triangle( v1, v2, v3 );
//    BBox bbox = getBBox( triangle );
//    bbox.min = { bbox.min[0] < 0 ? 0 : bbox.min[0], bbox.min[1] < 0 ? 0 : bbox.min[1] };
//    bbox.max = { bbox.max[0] > canvas(0).getW() ? canvas(0).getW() : bbox.max[0],
//                 bbox.max[1] > canvas(0).getH() ? canvas(0).getH() : bbox.max[1] };
//    for ( int x = bbox.min[0]; x < bbox.max[0]; x++ ) {
//        for (int y = bbox.min[1]; y < bbox.max[1]; y++) {
//            if (!isPointInTriangle(triangle, Vec2d(x, y))) continue;
//            double z = getZ(x, y, v1, v2, v3);
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
//    Vec3d v0 = transform( tri.v1 );
//    Vec3d v1 = transform( tri.v2 );
//    Vec3d v2 = transform( tri.v3 );
//// Sort vertices by y-coordinate
//    if (v0[1] > v1[1]) std::swap(v0, v1);
//    if (v0[1] > v2[1]) std::swap(v0, v2);
//    if (v1[1] > v2[1]) std::swap(v1, v2);
//
//// Compute inverse slopes
//    double invslope1 = (v1[0] - v0[0]) / static_cast<double>(v1[1] - v0[1]);
//    double invslope2 = (v2[0] - v0[0]) / static_cast<double>(v2[1] - v0[1]);
//    double invslope3 = (v2[0] - v1[0]) / static_cast<double>(v2[1] - v1[1]);
//
//// Starting x-coordinates
//    double curx1 = v0[0];
//    double curx2 = v0[0];
//
//// Fill bottom part of the triangle (v0 to v1)
//    double asd1 = v0[1] < 0 ? 0 : v0[1];
//    double asd2 = v1[1] > canvas(0).getH() ? canvas(0).getH() : v1[1];
//    for (int y = asd1; y < asd2; ++y) {
//        drawHorizontalLine(static_cast<int>(curx1), static_cast<int>(curx2), y, color, &canvas(0) );
//        curx1 += invslope1;
//        curx2 += invslope2;
//    }
//
//// Fill top part of the triangle (v1 to v2)
//    curx1 = v1[0];
//    asd1 = v1[1] < 0 ? 0 : v1[1];
//    asd2 = v2[1] > canvas(0).getH() ? canvas(0).getH() : v2[1];
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
    for ( const auto& triangle:mesh->getPrimitives() ) {
        rasterizer->drawFilledTriangle( *(Triangle*)triangle, mesh->getMaterial().getColor() );
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
