//
// Created by auser on 6/12/25.
//

#ifndef VULKAN_MODELMESH_H
#define VULKAN_MODELMESH_H

#include <glm/glm.hpp>

namespace crv::model {
    struct Mesh {
        uint32_t numIndices = 0;
        uint32_t numVertices = 0;
        uint32_t baseVertex = 0;
        uint32_t baseIndex = 0;
        uint32_t validFaces = 0;
        int materialIndex = -1;
        glm::mat4 transformation;
    };
}

#endif //VULKAN_MODELMESH_H
