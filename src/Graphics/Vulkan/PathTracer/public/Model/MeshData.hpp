    //
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_MESHDATA_HPP
#define COLLECTION_MESHDATA_HPP

#include <string>
#include <vector>

#include "SharedTypes.h"
#include "BBox.hpp"

namespace crv::graphics::vulkan {
    struct MeshData {
        std::vector<Vertex>   vertices{};
        std::vector<uint32_t> indices{};
        std::vector<float>    triAreas{};
        float                 area       = 0.0f;
        BBox<float>           bbox{};
        uint32_t              indexCount = 0;
        uint32_t              modelIndex = 0;
        std::string           meshName{};
    };
}

#endif //COLLECTION_MESHDATA_HPP
