//
// Created by igor on 14.01.2024.
//

#include "Light.h"
#include "RGB.h"

Light::Light():intensity( 0 ) {
    lightColor = RGB(255,255,255);
}


Light::Type Light::getType() const {
    return BASE;
}