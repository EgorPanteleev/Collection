#include "Material.h"
Material::Material(): color(), texture(), intensity( 0 ), diffuse( 0 ), roughness( 1 ), metalness( 0 ) {
}

Material::Material( const RGB& color, float intensity ): color( color ), texture(), intensity( intensity ) {}
Material::Material( const RGB& color, float diffuse, float roughness ): color( color ), texture(), intensity( 0 ), diffuse( diffuse ), roughness( roughness ) {
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

float Material::getRoughness() const {
    return roughness;
}
void Material::setRoughness( float r ) {
    roughness = r;
}

float Material::getMetalness() const {
    return metalness;
}
void Material::setMetalness( float m ) {
    metalness = m;
}

void Material::setTexture( const std::string& path ) {
    texture = { path };
}
Texture Material::getTexture() const {
    return texture;
}
