//
// Created by igor on 9/22/26.
//

#ifndef COLLECTION_FOG_HPP
#define COLLECTION_FOG_HPP

#include <glm/glm.hpp>

namespace crv::graphics::vulkan {
    struct Fog {
        glm::vec3 color{1.0f};
        float     density    = 0.0f;
        float     height     = 5.0f;
        float     base       = 0.0f;
        float     anisotropy = 0.0f;
    };
}

#endif //COLLECTION_FOG_HPP
