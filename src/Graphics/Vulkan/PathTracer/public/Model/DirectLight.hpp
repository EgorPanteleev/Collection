//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_DIRECTLIGHT_HPP
#define COLLECTION_DIRECTLIGHT_HPP

#include <glm/glm.hpp>

namespace crv::graphics::vulkan {
    struct DirectLight {
        glm::vec3 dir{};
        float     intensity = 0;
    };
}

#endif //COLLECTION_DIRECTLIGHT_HPP
