//
// Created by auser on 6/18/25.
//

#ifndef VULKAN_BBOX_H
#define VULKAN_BBOX_H

#include "glm/glm.hpp"

class BBox {
public:
    BBox(): min(FLT_MAX), max(-FLT_MAX) {}

    glm::vec3 min;
    glm::vec3 max;
};

#endif //VULKAN_BBOX_H
