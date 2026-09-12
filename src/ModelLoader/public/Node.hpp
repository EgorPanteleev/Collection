//
// Created by igor on 6/12/26.
//

#ifndef VULKAN_NODE_HPP
#define VULKAN_NODE_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace crv::model {
    struct Node {
        std::string           name{};
        glm::mat4             transform{1.0f};
        std::vector<uint32_t> meshes{};
        std::vector<Node>     children{};
    };
}

#endif //VULKAN_NODE_HPP
