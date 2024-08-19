#pragma once
#include "Color.h"
#include "string"
#include "stb_image.h"
#include "iostream"
class ImageData {
public:
    ImageData(): data(nullptr), width( 0 ), height( 0 ), channels( 0 ) {}
    ImageData( const std::string& path ) {
        data = stbi_load( path.c_str(), &width, &height, &channels, 0);
        if (!data) std::cerr << "Error loading texture: " << path << std::endl;
    }
    unsigned char* data;
    int width;
    int height;
    int channels;
};

class Texture {
public:
    Texture (): colorMap(), normalMap(), ambientMap(), roughnessMap() {}
    Texture ( const std::string& path ) {
        colorMap = ImageData( path + "Color.jpg" );
        if ( !colorMap.data ) colorMap = ImageData( path + "Color.png" );
        normalMap = ImageData( path + "NormalDX.jpg" );
        if ( !normalMap.data ) normalMap = ImageData( path + "NormalDX.png" );
        ambientMap = ImageData( path + "AmbientOcclusion.jpg" );
        if ( !ambientMap.data ) ambientMap = ImageData( path + "AmbientOcclusion.png" );
        roughnessMap = ImageData( path + "Roughness.jpg" );
        if ( !roughnessMap.data ) roughnessMap = ImageData( path + "Roughness.png" );
    }
    ImageData colorMap;
    ImageData normalMap;
    ImageData ambientMap;
    ImageData roughnessMap;
};



class Material {
public:
    Material();
    Material( const RGB& color, float intensity );
    Material( const RGB& color, float diffuse, float reflection );
    [[nodiscard]] RGB getColor() const;
    void setColor( const RGB& c );
    [[nodiscard]] float getIntensity() const;
    void setIntensity( float i );
    [[nodiscard]] float getDiffuse() const;
    void setDiffuse( float d );
    [[nodiscard]] float getReflection() const;
    void setReflection( float r );
    void setTexture( const std::string& path );
    [[nodiscard]] Texture getTexture() const;
private:
    RGB color;
    Texture texture;
    float intensity;
    float diffuse;
    float reflection;
};

