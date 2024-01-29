#include "Material.h"

Material::Material(): color(), diffuse( 0 ), reflection( 0 ) {
}
Material::Material( const RGB& color, float diffuse, float reflection ): color( color ) ,diffuse( diffuse ), reflection( reflection ) {
}

RGB Material::getColor() const {
    return color;
}
void Material::setColor( const RGB& c ) {
    color = c;
}

float Material::getDiffuse() const {
    return diffuse;
}
void Material::setDiffuse( float d ) {
    diffuse = d;
}

float Material::getReflection() const {
    return reflection;
}
void Material::setReflection( float r ) {
    reflection = r;
}