#include "Material.h"

__host__ __device__ Material::Material(): color(), diffuse( 0 ), reflection( 0 ) {
}
__host__ __device__ Material::Material( const RGB& color, float diffuse, float reflection ): color( color ) ,diffuse( diffuse ), reflection( reflection ) {
}

__host__ __device__ RGB Material::getColor() const {
    return color;
}
__host__ __device__ void Material::setColor( const RGB& c ) {
    color = c;
}

__host__ __device__ float Material::getDiffuse() const {
    return diffuse;
}
__host__ __device__ void Material::setDiffuse( float d ) {
    diffuse = d;
}

__host__ __device__ float Material::getReflection() const {
    return reflection;
}
__host__ __device__ void Material::setReflection( float r ) {
    reflection = r;
}