//
// Created by igor on 07.01.2024.
//
#include "Shape.h"

RGB Shape::getColor() const {
    return color;
}
void Shape::setColor( const RGB& c ) {
    color = c;
}