#include "Material.h"
Material::Material(): color(), texture(), intensity( 0 ), diffuse( 0 ), roughness( 1 ), metalness( 0 ) {
}

Material::Material( const RGB& color, double intensity ):
    color( color ), texture(), intensity( intensity ) {}
Material::Material( const RGB& color, double diffuse, double roughness ):
    color( color ), texture(), intensity( 0 ), diffuse( diffuse ), roughness( roughness ), metalness( 0 ) {
}
Material::Material( const RGB& color, double diffuse, double roughness, double metalness ):
    color( color ), texture(), intensity( 0 ), diffuse( diffuse ), roughness( roughness ), metalness( metalness ) {}


RGB Material::getColor() const {
    return color;
}

void Material::setColor( const RGB& c ) {
    color = c;
}

double Material::getIntensity() const {
    return intensity;
}

void Material::setIntensity( double i ) {
    intensity = i;
}

double Material::getDiffuse() const {
    return diffuse;
}
void Material::setDiffuse( double d ) {
    diffuse = d;
}

double Material::getRoughness() const {
    return roughness;
}
void Material::setRoughness( double r ) {
    roughness = r;
}

double Material::getMetalness() const {
    return metalness;
}
void Material::setMetalness( double m ) {
    metalness = m;
}

void Material::setTexture( const std::string& path ) {
    texture = { path };
}
Texture Material::getTexture() const {
    return texture;
}
