//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_SELECTION_HPP
#define COLLECTION_SELECTION_HPP

#include <cstdint>
#include <vector>

namespace crv::graphics::vulkan {
    struct Selection {
        std::vector<uint32_t> selectedInstances{};
        uint32_t              activeInstance = UINT32_MAX;
        bool                  pending        = false;
    };
}

#endif //COLLECTION_SELECTION_HPP
