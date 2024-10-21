#pragma once
#include "RGB.h"
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
    Texture (): colorMap(), normalMap(), ambientMap(), roughnessMap(), metalnessMap() {}
    Texture ( const std::string& path ) {
        colorMap = ImageData( path + "Color.jpg" );
        if ( !colorMap.data ) colorMap = ImageData( path + "Color.png" );
        normalMap = ImageData( path + "NormalDX.jpg" );
        if ( !normalMap.data ) normalMap = ImageData( path + "NormalDX.png" );
        ambientMap = ImageData( path + "AmbientOcclusion.jpg" );
        if ( !ambientMap.data ) ambientMap = ImageData( path + "AmbientOcclusion.png" );
        roughnessMap = ImageData( path + "Roughness.jpg" );
        if ( !roughnessMap.data ) roughnessMap = ImageData( path + "Roughness.png" );
        metalnessMap = ImageData( path + "Metalness.jpg" );
        if ( !metalnessMap.data ) metalnessMap = ImageData( path + "Metalness.png" );
    }
    ImageData colorMap;
    ImageData normalMap;
    ImageData ambientMap;
    ImageData roughnessMap;
    ImageData metalnessMap;
};



class Material {
public:
    Material();
    Material( const RGB& color, double intensity );
    Material( const RGB& color, double diffuse, double roughness );
    [[nodiscard]] RGB getColor() const;
    void setColor( const RGB& c );
    [[nodiscard]] double getIntensity() const;
    void setIntensity( double i );
    [[nodiscard]] double getDiffuse() const;
    void setDiffuse( double d );
    [[nodiscard]] double getRoughness() const;
    void setRoughness( double r );
    double getMetalness() const;
    void setMetalness( double m );
    void setTexture( const std::string& path );
    [[nodiscard]] Texture getTexture() const;
private:
    RGB color;
    Texture texture;
    double intensity;
    double diffuse;
    double roughness;
    double metalness;
};

