#include "Material.h"
Material::Material(): color(), texture(), intensity( 0 ), diffuse( 0 ), reflection( 0 ) {
}

Material::Material( const RGB& color, float intensity ): color( color ), texture(), intensity( intensity ) {}
Material::Material( const RGB& color, float diffuse, float reflection ): color( color ), texture(), intensity( 0 ), diffuse( diffuse ), reflection( reflection ) {
}

RGB Material::getColor() const {
    return color;
}

void Material::setColor( const RGB& c ) {
    color = c;
}

float Material::getIntensity() const {
    return intensity;
}

void Material::setIntensity( float i ) {
    intensity = i;
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

void Material::setTexture( const std::string& path ) {
    texture = { path };
}
Texture Material::getTexture() const {
    return texture;
}
