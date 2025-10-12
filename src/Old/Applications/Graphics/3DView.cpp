#include "iostream"
#include "Vec3.h"
#include "Ray.h"
#include <cmath>
#include <algorithm>


double Lambda(const Vec3d& wo, const Vec2d& a)
{
    return (-1.0 + std::sqrt(1.0 + (pow(a.x * wo.x, 2) + pow(a.y * wo.y, 2)) / (pow(wo.z, 2)))) / 2.0;
}


double SmithG1(const Vec3d& wo, const Vec2d& a)
{
    return 1.0 / (1.0 + Lambda(wo, a));
}

double D(const Vec3d& m, const Vec2d& a)
{
    return 1.0 / (M_PI * a.x * a.y * pow(pow(m.x / a.x,2) + pow(m.y / a.y,2) + pow(m.z,2), 2));
}

double DV(const Vec3d& m, const Vec3d& wo, const Vec2d& a)
{
    return SmithG1(wo, a) * dot(wo, m) * D(m, a) / wo.z;
}


double SmithG2(const Vec3d& wi, const Vec3d& wo, const Vec2d& a)
{
    return 1.0 / (1.0 + Lambda(wo, a) + Lambda(wi, a));
}

double reflection(const Vec3d& wi, const Vec3d& wo, Vec2d& a, double& PDF)
{
    Vec3d m = (wo + wi).normalize();

    PDF = DV(m, wo, a) / (4.0 * dot(m, wo));
    return  D(m, a) * SmithG2(wi, wo, a) / (4.0 * wo.z * wi.z);
}
Vec3d visibleMicrofacet( const Vec3d& wo, const Vec2d& a )
{
    double u = randomdouble();
    double v = randomdouble();
    Vec3d Vh = (Vec3d(a.x * wo.x, a.y * wo.y, wo.z)).normalize();

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    double len2 = pow(Vh.x, 2) + pow(Vh.y, 2);
    Vec3d T1 = len2 > 0.0 ? Vec3d(-Vh.y, Vh.x, 0.0) * 1.0 / (double) sqrt(len2) : Vec3d(1.0, 0.0, 0.0);
    Vec3d T2 = Vh.cross( T1 );

    // Section 4.2: parameterization of the projected area
    double r = std::sqrt(u);
    double phi = v * 2 * M_PI;
    double t1 = r * std::cos(phi);
    double t2 = r * std::sin(phi);
    double s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * std::sqrt(1.0 - pow(t1, 2)) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    Vec3d Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max(0.0, 1.0 - pow(t1, 2) - pow(t2, 2))) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return (Vec3d(a.x * Nh.x, a.y * Nh.y, std::max(0.0, Nh.z))).normalize();
}



Vec3d reflect( const Vec3d& wo, const Vec3d& N ) {
    return  ( wo - N * 2 * dot(N, wo ) ).normalize();
}

Vec3d GGXVNDFSample( Vec3d wi, double roughness, Vec3d N ) {
    // decompose the doubletor in parallel and perpendicular components
    Vec2d u = { randomdouble(), randomdouble() };

    Vec3d wi_z = (-1) * N * dot(wi, N);
    Vec3d wi_xy = wi + wi_z;

    // warp to the hemisphere configuration
    Vec3d wiStd = (-1)*(roughness * wi_xy + wi_z).normalize();

    // sample a spherical cap in (-wiStd.z, 1]
    double wiStd_z = dot(wiStd, N);
    double z = 1.0 - u.y * (1.0 + wiStd_z);
    double sinTheta = sqrt(std::clamp(1.0 - z * z, 0.0, 1.0));
    double phi = M_PI * 2 * u.x - M_PI;
    double x = sinTheta * cos(phi);
    double y = sinTheta * sin(phi);
    Vec3d cStd = Vec3d(x, y, z);

    // reflect sample to align with normal
    Vec3d up = Vec3d(0, 0, 1.000001); // Used for the singularity
    Vec3d wr = N + up;
    Vec3d c = dot(wr, cStd) * wr / wr.z - cStd;

    // compute halfway direction as standard normal
    Vec3d wmStd = c + wiStd;
    Vec3d wmStd_z = N * dot(N, wmStd);
    Vec3d wmStd_xy = wmStd_z - wmStd;

    // return final normal
    return (roughness * wmStd_xy + wmStd_z).normalize();

}

Vec3d sampleGGXVNDF(Vec3d Ve, double alpha_x, double alpha_y)
{
    double U1 = randomdouble();
    double U2 = randomdouble();
// Section 3.2: transforming the view direction to the hemisphere configuration
    Vec3d Vh = (Vec3d(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z)).normalize();
// Section 4.1: orthonormal basis (with special case if cross product is zero)
    double lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    Vec3d T1 = lensq > 0 ? Vec3d(-Vh.y, Vh.x, 0) * 1.0 / sqrt(lensq) : Vec3d(1,0,0);
    Vec3d T2 = Vh.cross( T1);
// Section 4.2: parameterization of the projected area
    double r = sqrt(U1);
    double phi = 2.0 * M_PI * U2;
    double t1 = r * cos(phi);
    double t2 = r * sin(phi);
    double s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
// Section 4.3: reprojection onto hemisphere
    Vec3d Nh = t1*T1 + t2*T2 + sqrt(std::max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
// Section 3.4: transforming the normal back to the ellipsoid configuration
    Vec3d Ne = (Vec3d(alpha_x * Nh.x, alpha_y * Nh.y, std::max<double>(0.0, Nh.z))).normalize();
    return Ne;
}


void test() {
    double roughness = 0.001;
    int numSamples = 1;
    Vec3d P = { 0, 0, 0 };
    //0.957437, -0.233633, 0.169498
    Vec3d N1 = {  0, 0, -1};
    N1 = N1.normalize();
    Vec3d N2 = { 0, 0, 1 };
    N2 = N2.normalize();

    double dotProduct = dot( N1, N2 );

    double angle = acos(dotProduct) * 180 / M_PI;
    Mat3d rot;
    if (fabs(dotProduct + 1.0) < 1e-6) {
        // Vectors are opposite, choose a perpendicular axis
        Vec3d arbitraryAxis(1.0, 0.0, 0.0); // Choose (1, 0, 0) as an arbitrary axis
        rot = Mat3d::getRotationMatrix(arbitraryAxis, 180);
    } else {
        // Calculate the rotation axis and angle
        Vec3d axis = N2.cross( N1 ).normalize();
        rot = Mat3d::getRotationMatrix( axis, angle);
    }


    // Угол между N1 и N2 (через скалярное произведение)

    for (int j = 0; j < numSamples; j++ ) {
        Vec3d wo = { 0, 0, 1};
        Vec2d roughV = Vec2d( roughness, roughness);
        Vec3d N = sampleGGXVNDF( wo, roughness, roughness ); //Importance sampler, that returns N
       // N = rot * N;




        std::cout << " N = " << N << std::endl;
        Vec3d wi = reflect( wo, N ) * ( -1 );//to light
        std::cout << " wi = " << wi << std::endl;
        double PDF = 1.0;//pdf_vndf_isotropic( wo, wi, roughness, N );
        Vec2d rVec = { roughness, roughness };
        double refl = 1.0;//reflection( wi, wo, rVec , PDF );
        //double refl = reflection(wi, wo, vRoughness, PDF);
        Ray sampleRay = { P + wi * 1e-3, wi };
        double dNL = dot( wi, N );
        if ( dNL < 0 ) dNL = 0;
        //ambient = ambient + refl / PDF * traceRay( sampleRay, nextDepth - 1, throughput ).color * dNL;
        //ambient = ambient / PDF;
    }
}




int main(int argc, char* argv[]) {
    srand(time( nullptr ));
    test();
    return 0;

}

