//
// Created by auser on 10/21/24.
//
//
// Created by auser on 10/21/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include "Sampler.h"
#include "CoordinateSystem.h"
#define EPSILON 1e-5


TEST( CosineWeighted, TestPDF ) {
    EXPECT_NEAR( CosineWeighted::PDF( 0.5 ), 0.27934, EPSILON );

    EXPECT_NEAR( CosineWeighted::PDF( 12 ), 0.2686, EPSILON );

    EXPECT_NEAR(CosineWeighted::PDF(0.0), M_1_PI, EPSILON);

    EXPECT_NEAR(CosineWeighted::PDF(M_PI / 2.0), 0.0, EPSILON);

    EXPECT_NEAR(CosineWeighted::PDF(3 * M_PI / 4.0), std::cos(3 * M_PI / 4.0) * M_1_PI, EPSILON);

    EXPECT_NEAR(CosineWeighted::PDF(M_PI), -M_1_PI, EPSILON);

    Vec3d N(0.0, 0.0, 1.0);
    Vec3d wi(0.0, 0.0, 1.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), 0.171983, EPSILON);

    wi = Vec3d(1.0, 0.0, 0.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), M_1_PI, EPSILON);

    wi = Vec3d(0.0, 0.0, -1.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), 0.171983, EPSILON);

    wi = Vec3d(1.0, 1.0, 1.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), 0.171983, EPSILON);

    N = Vec3d(1.0, 0.0, 0.0);
    wi = Vec3d(1.0, 0.0, 0.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), 0.171983, EPSILON);

    wi = Vec3d(1.0, 1.0, 0.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), 0.171983, EPSILON);

    N = Vec3d(1.0, 1.0, 1.0);
    wi = Vec3d(-1.0, -1.0, -1.0);
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), -0.315124399, EPSILON);

    N = Vec3d(3.0, 4.0, 0.0).normalize();
    wi = Vec3d(-4.0, 3.0, 0.0).normalize();
    EXPECT_NEAR(CosineWeighted::PDF(N, wi), M_1_PI, EPSILON);
}

TEST( Lambertian, TestBRDF ) {
    EXPECT_NEAR( Lambertian::BRDF(), M_1_PI, EPSILON );
}

TEST( OrenNayar, TestBRDF ) {
    Vec3d wi = { 0, 0.5, 1 };
    Vec3d wo = { 0, 0.5, 0.2 };
    double alpha = 0.5;
    EXPECT_NEAR( OrenNayar::BRDF( wi, wo, alpha ), 0.22243341, EPSILON );
    wi = { 1, 0.5, 1 };
    wo = { 1, 0.5, 0.2 };
    alpha = 0.3;
    EXPECT_NEAR( OrenNayar::BRDF( wi, wo, alpha ), 0.242521792, EPSILON );
    wi = { 1, 12, 1 };
    wo = { 1, 0.5, 0.2 };
    alpha = 0.3;
    EXPECT_NEAR( OrenNayar::BRDF( wi, wo, alpha ), 0.242521792, EPSILON );
    wi = { 1, 12, 1 };
    wo = { 1, 0.5, 12 };
    alpha = 0.7;
    EXPECT_NEAR( OrenNayar::BRDF( wi, wo, alpha ), 0.210146322, EPSILON );
}

TEST( GGX, TestF ) {
    EXPECT_NEAR( GGX::F( 0.5 ), 0.90312498, EPSILON );
    EXPECT_NEAR( GGX::F( 0.7 ), 0.90024298, EPSILON );
    EXPECT_NEAR( GGX::F( 0.3 ), 0.91680699, EPSILON );
    EXPECT_NEAR( GGX::F( 0.23 ), 0.92706781, EPSILON );
    EXPECT_NEAR( GGX::F( 1.91 ), 0.83759677, EPSILON );
    EXPECT_NEAR( GGX::F( 15.345 ), -60743.012727, EPSILON );
}

TEST( GGX, TestD ) {
    Vec2d a = { 0.5, 0.5 };
    EXPECT_NEAR( GGX::D( { 1, 0.2, 0.3 }, a ), 0.07049077, EPSILON );
    EXPECT_NEAR( GGX::D( { 1, 0.2, 1.3 }, a ), 0.03720475, EPSILON );
    EXPECT_NEAR( GGX::D( { 1.33, 0.76, 0.3 }, a ), 0.01417946, EPSILON );
    EXPECT_NEAR( GGX::D( { 13.5, 12.2, 0.3 }, a ), 0, EPSILON );
    EXPECT_NEAR( GGX::D( { 103, 1.2, 0.99 }, a ), 0, EPSILON );
    EXPECT_NEAR( GGX::D( { 1, 0.2, 17.3 }, a ), 1.38272525e-5, EPSILON );
}

TEST( GGX, TestLambda ) {
    Vec2d a = { 0.5, 0.5 };
    EXPECT_NEAR( GGX::Lambda( { 1, 0.2, 0.3 }, a ), 0.48601323, EPSILON );
    EXPECT_NEAR( GGX::Lambda( { 1, 0.2, 1.3 }, a ), 0.03708612, EPSILON );
    EXPECT_NEAR( GGX::Lambda( { 1.33, 0.76, 0.3 }, a ), 0.8709536, EPSILON );
    EXPECT_NEAR( GGX::Lambda( { 13.5, 12.2, 0.3 }, a ), 14.671473, EPSILON );
    EXPECT_NEAR( GGX::Lambda( { 103, 1.2, 0.99 }, a ), 25.516672, EPSILON );
    EXPECT_NEAR( GGX::Lambda( { 1, 0.2, 17.3 }, a ), 0.00021713, EPSILON );
}

TEST( GGX, TestG1) {
    Vec2d a = { 0.5, 0.5 };
    EXPECT_NEAR( GGX::SmithG1( { 1, 0.2, 0.3 }, a ), 0.6729415, EPSILON );
    EXPECT_NEAR( GGX::SmithG1( { 1, 0.2, 1.3 }, a ), 0.9642400, EPSILON );
    EXPECT_NEAR( GGX::SmithG1( { 1.33, 0.76, 0.3 }, a ), 0.5344867, EPSILON );
    EXPECT_NEAR( GGX::SmithG1( { 13.5, 12.2, 0.3 }, a ), 0.0638102, EPSILON );
    EXPECT_NEAR( GGX::SmithG1( { 103, 1.2, 0.99 }, a ), 0.037712, EPSILON );
    EXPECT_NEAR( GGX::SmithG1( { 1, 0.2, 17.3 }, a ), 0.9997828, EPSILON );
}

TEST( GGX, TestDV) {
    Vec2d a = { 0.5, 0.5 };
    EXPECT_NEAR( GGX::DV( { 1, 0.2, 0.3 }, { 1, 0.3, 0.3 }, a ), 0.17972774, EPSILON );
    EXPECT_NEAR( GGX::DV( { 1, 0.2, 1.3 }, { 0.15, 0.2, 0.3 }, a ), 0.06905202, EPSILON );
    EXPECT_NEAR( GGX::DV( { 1.33, 0.76, 0.3 }, { 13, 0.2, 0.3 }, a ), 0.07303345, EPSILON );
    EXPECT_NEAR( GGX::DV( { 13.5, 12.2, 0.3 }, { 1.24, 0.2, 1.3 }, a ), 1.0354188e-5, EPSILON );
    EXPECT_NEAR( GGX::DV( { 103, 1.2, 0.99 }, { 1, 0.2, 0.3 }, a ), 1.641552955e-07, EPSILON );
    EXPECT_NEAR( GGX::DV( { 1, 0.2, 17.3 }, { 1.98, 0.2, 0.3 }, a ), 0.0001488778, EPSILON );
}

TEST( GGX, TestG2) {
    Vec2d a = { 0.5, 0.5 };
    EXPECT_NEAR( GGX::SmithG2( { 1, 0.2, 0.3 }, { 1, 0.3, 0.3 }, a ), 0.5026440, EPSILON );
    EXPECT_NEAR( GGX::SmithG2( { 1, 0.2, 1.3 }, { 0.15, 0.2, 0.3 }, a ), 0.9269963, EPSILON );
    EXPECT_NEAR( GGX::SmithG2( { 1.33, 0.76, 0.3 }, { 13, 0.2, 0.3 }, a ), 0.0818524, EPSILON );
    EXPECT_NEAR( GGX::SmithG2( { 13.5, 12.2, 0.3 }, { 1.24, 0.2, 1.3 }, a ), 0.06358588, EPSILON );
    EXPECT_NEAR( GGX::SmithG2( { 103, 1.2, 0.99 }, { 1, 0.2, 0.3 }, a ), 0.0370333, EPSILON );
    EXPECT_NEAR( GGX::SmithG2( { 1, 0.2, 17.3 }, { 1.98, 0.2, 0.3 }, a ), 0.4479587, EPSILON );
}

TEST( GGX, TestPDFBRDF) {
    Vec2d a = { 0.5, 0.5 };
    double PDF;
    EXPECT_NEAR( GGX::BRDF( { 1, 0.2, 0.3 }, { 1, 0.3, 0.3 }, a, PDF ), 0.1128269, EPSILON );
    EXPECT_NEAR( PDF, 0.0497665, EPSILON );
    EXPECT_NEAR( GGX::BRDF( { 1, 0.2, 1.3 }, { 0.15, 0.2, 0.3 }, a, PDF ), 0.156160354, EPSILON );
    EXPECT_NEAR( PDF, 0.23093231, EPSILON );
    EXPECT_NEAR( GGX::BRDF( { 1.33, 0.76, 0.3 }, { 13, 0.2, 0.3 }, a, PDF ), -448.66307918, EPSILON );
    EXPECT_NEAR( PDF, 0.0058599, EPSILON );
    EXPECT_NEAR( GGX::BRDF( { 13.5, 12.2, 0.3 }, { 1.24, 0.2, 1.3 }, a, PDF ), 0.00294940, EPSILON );
    EXPECT_NEAR( PDF, 0.0146518, EPSILON );
    EXPECT_NEAR( GGX::BRDF( { 103, 1.2, 0.99 }, { 1, 0.2, 0.3 }, a, PDF ), 0.002233107, EPSILON );
    EXPECT_NEAR( PDF, 0.04463613, EPSILON );
    EXPECT_NEAR( GGX::BRDF( { 1, 0.2, 17.3 }, { 1.98, 0.2, 0.3 }, a, PDF ), 0.0210173, EPSILON );
    EXPECT_NEAR( PDF, 0.4037304, EPSILON );
}

TEST( GGX, TestGetNormal) {
    Vec2d a = { 0.5, 0.5 };
    EXPECT_NEAR( GGX::getNormal( { 1, 0.2, 0.3 }, a )[0], 0.0911152, EPSILON );
    EXPECT_NEAR( GGX::getNormal( { 1, 0.2, 1.3 }, a )[0], 0.797648787, EPSILON );
    EXPECT_NEAR( GGX::getNormal( { 1.33, 0.76, 0.3 }, a )[0], -0.127637192, EPSILON );
    EXPECT_NEAR( GGX::getNormal( { 13.5, 12.2, 0.3 }, a )[0], 0.6345429420, EPSILON );
    EXPECT_NEAR( GGX::getNormal( { 103, 1.2, 0.99 }, a )[0], 0.682546913, EPSILON );
    EXPECT_NEAR( GGX::getNormal( { 1, 0.2, 17.3 }, a )[0], 0.38196161, EPSILON );
}

void checkCS( const Vec3d& vec, const Vec3d& to, const Vec3d& from ) {
    CoordinateSystem cs( vec );
    Vec3d w = { 0.23, 0.57, 0.91 };
    Vec3d wTo = cs.to( w );
    Vec3d wFr = cs.from( w );
    EXPECT_NEAR( wTo[0], to[0], EPSILON );
    EXPECT_NEAR( wTo[1], to[1], EPSILON );
    EXPECT_NEAR( wTo[2], to[2], EPSILON );
    EXPECT_NEAR( wFr[0], from[0], EPSILON );
    EXPECT_NEAR( wFr[1], from[1], EPSILON );
    EXPECT_NEAR( wFr[2], from[2], EPSILON );
}

TEST( CoordinateSystem, test1) {
    checkCS( { 0, 0, 1 }, { 0.23, 0.57, 0.91 }, { 0.23, 0.57, 0.91 } );
    checkCS( { 0, 1, 0 }, { 0.23, -0.91, 0.57 }, { 0.23, 0.91, -0.57 } );
    checkCS( { 1, 0, 0 }, { -0.91, 0.57, 0.23 }, { 0.91, 0.57, -0.23 } );
    checkCS( { 0.2, 0.3, 1.2 }, { 0.0282727, 0.267409, 1.309 }, { 0.39227, 0.813409, 0.8750001 } );
    checkCS( { 1.2, 0.2, 0.3 }, { -1.222, 0.328, 0.663 }, { 0.962, 0.692, -0.117 } );
    checkCS( { 0.3, 1.2, 0.2 }, { -0.23125, -1.275, 0.935 }, { 0.31475, 0.909, -0.571 } );
    checkCS( { 12, 17, 0.347 }, { -121.603134, -172.026947, 12.76577 }, { -99.7631378, -141.0869445, -12.13422966 } );
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
//    ::testing::GTEST_FLAG( color ) = "yes";
    return RUN_ALL_TESTS();
}
