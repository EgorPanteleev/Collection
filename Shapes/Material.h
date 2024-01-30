#pragma once
#include "Color.h"
class Material {
public:
    Material();
    Material( const RGB& color, float diffuse, float reflection );
    [[nodiscard]] RGB getColor() const;
    void setColor( const RGB& c );
    [[nodiscard]] float getDiffuse() const;
    void setDiffuse( float d );
    [[nodiscard]] float getReflection() const;
    void setReflection( float r );
private:
    RGB color;
    float diffuse;
    float reflection;
};

