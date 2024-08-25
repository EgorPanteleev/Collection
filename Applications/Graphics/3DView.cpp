#include "iostream"
#include "Vector3f.h"
#include "Ray.h"
#include "Utils.h"
#include <cmath>
#include <algorithm>


double Lambda(const Vector3f& wo, const Vector2f& a)
{
    return (-1.0 + std::sqrt(1.0 + (pow(a.x * wo.x, 2) + pow(a.y * wo.y, 2)) / (pow(wo.z, 2)))) / 2.0;
}


double SmithG1(const Vector3f& wo, const Vector2f& a)
{
    return 1.0 / (1.0 + Lambda(wo, a));
}

double D(const Vector3f& m, const Vector2f& a)
{
    return 1.0 / (M_PI * a.x * a.y * pow(pow(m.x / a.x,2) + pow(m.y / a.y,2) + pow(m.z,2), 2));
}

double DV(const Vector3f& m, const Vector3f& wo, const Vector2f& a)
{
    return SmithG1(wo, a) * dot(wo, m) * D(m, a) / wo.z;
}


double SmithG2(const Vector3f& wi, const Vector3f& wo, const Vector2f& a)
{
    return 1.0 / (1.0 + Lambda(wo, a) + Lambda(wi, a));
}

double reflection(const Vector3f& wi, const Vector3f& wo, Vector2f& a, double& PDF)
{
    Vector3f m = (wo + wi).normalize();

    PDF = DV(m, wo, a) / (4.0 * dot(m, wo));
    return  D(m, a) * SmithG2(wi, wo, a) / (4.0 * wo.z * wi.z);
}
Vector3f visibleMicrofacet( const Vector3f& wo, const Vector2f& a )
{
    float u = randomFloat();
    float v = randomFloat();
    Vector3f Vh = (Vector3f(a.x * wo.x, a.y * wo.y, wo.z)).normalize();

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    double len2 = pow(Vh.x, 2) + pow(Vh.y, 2);
    Vector3f T1 = len2 > 0.0 ? Vector3f(-Vh.y, Vh.x, 0.0) * 1.0f / (float) sqrt(len2) : Vector3f(1.0, 0.0, 0.0);
    Vector3f T2 = Vh.cross( T1 );

    // Section 4.2: parameterization of the projected area
    double r = std::sqrt(u);
    double phi = v * 2 * M_PI;
    double t1 = r * std::cos(phi);
    double t2 = r * std::sin(phi);
    double s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * std::sqrt(1.0 - pow(t1, 2)) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    Vector3f Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max(0.0, 1.0 - pow(t1, 2) - pow(t2, 2))) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return (Vector3f(a.x * Nh.x, a.y * Nh.y, std::max(0.0f, Nh.z))).normalize();
}



Vector3f reflect( const Vector3f& wo, const Vector3f& N ) {
    return  ( wo - N * 2 * dot(N, wo ) ).normalize();
}

Vector3f GGXVNDFSample( Vector3f wi, float roughness, Vector3f N ) {
    // decompose the floattor in parallel and perpendicular components
    Vector2f u = { randomFloat(), randomFloat() };

    Vector3f wi_z = (-1) * N * dot(wi, N);
    Vector3f wi_xy = wi + wi_z;

    // warp to the hemisphere configuration
    Vector3f wiStd = (-1)*(roughness * wi_xy + wi_z).normalize();

    // sample a spherical cap in (-wiStd.z, 1]
    float wiStd_z = dot(wiStd, N);
    float z = 1.0 - u.y * (1.0 + wiStd_z);
    float sinTheta = sqrt(std::clamp(1.0f - z * z, 0.0f, 1.0f));
    float phi = M_PI * 2 * u.x - M_PI;
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    Vector3f cStd = Vector3f(x, y, z);

    // reflect sample to align with normal
    Vector3f up = Vector3f(0, 0, 1.000001); // Used for the singularity
    Vector3f wr = N + up;
    Vector3f c = dot(wr, cStd) * wr / wr.z - cStd;

    // compute halfway direction as standard normal
    Vector3f wmStd = c + wiStd;
    Vector3f wmStd_z = N * dot(N, wmStd);
    Vector3f wmStd_xy = wmStd_z - wmStd;

    // return final normal
    return (roughness * wmStd_xy + wmStd_z).normalize();

}

Vector3f sampleGGXVNDF(Vector3f Ve, float alpha_x, float alpha_y)
{
    float U1 = randomFloat();
    float U2 = randomFloat();
// Section 3.2: transforming the view direction to the hemisphere configuration
    Vector3f Vh = (Vector3f(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z)).normalize();
// Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    Vector3f T1 = lensq > 0 ? Vector3f(-Vh.y, Vh.x, 0) * 1.0f / sqrt(lensq) : Vector3f(1,0,0);
    Vector3f T2 = Vh.cross( T1);
// Section 4.2: parameterization of the projected area
    float r = sqrt(U1);
    float phi = 2.0 * M_PI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
// Section 4.3: reprojection onto hemisphere
    Vector3f Nh = t1*T1 + t2*T2 + sqrt(std::max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
// Section 3.4: transforming the normal back to the ellipsoid configuration
    Vector3f Ne = (Vector3f(alpha_x * Nh.x, alpha_y * Nh.y, std::max<float>(0.0, Nh.z))).normalize();
    return Ne;
}


void test() {
    float roughness = 0.001;
    int numSamples = 1;
    Vector3f P = { 0, 0, 0 };
    //0.957437, -0.233633, 0.169498
    Vector3f N1 = {  0, 0, -1};
    N1 = N1.normalize();
    Vector3f N2 = { 0, 0, 1 };
    N2 = N2.normalize();

    double dotProduct = dot( N1, N2 );

    double angle = acos(dotProduct) * 180 / M_PI;
    Mat3f rot;
    if (fabs(dotProduct + 1.0f) < 1e-6) {
        // Vectors are opposite, choose a perpendicular axis
        Vector3f arbitraryAxis(1.0, 0.0, 0.0); // Choose (1, 0, 0) as an arbitrary axis
        rot = Mat3f::getRotationMatrix(arbitraryAxis, 180);
    } else {
        // Calculate the rotation axis and angle
        Vector3f axis = N2.cross( N1 ).normalize();
        rot = Mat3f::getRotationMatrix( axis, angle);
    }


    // Угол между N1 и N2 (через скалярное произведение)

    for (int j = 0; j < numSamples; j++ ) {
        Vector3f wo = { 0, 0, 1};
        Vector2f roughV = Vector2f( roughness, roughness);
        Vector3f N = sampleGGXVNDF( wo, roughness, roughness ); //Importance sampler, that returns N
       // N = rot * N;




        std::cout << " N = " << N << std::endl;
        Vector3f wi = reflect( wo, N ) * ( -1 );//to light
        std::cout << " wi = " << wi << std::endl;
        float PDF = 1.0f;//pdf_vndf_isotropic( wo, wi, roughness, N );
        Vector2f rVec = { roughness, roughness };
        float refl = 1.0f;//reflection( wi, wo, rVec , PDF );
        //double refl = reflection(wi, wo, vRoughness, PDF);
        Ray sampleRay = { P + wi * 1e-3, wi };
        float dNL = dot( wi, N );
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

