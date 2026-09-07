//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_UPDATESTATE_HPP
#define COLLECTION_UPDATESTATE_HPP

#include <cstdint>
#include <vector>

namespace crv::graphics::vulkan {
    enum class InstanceUpdate { Data, Model };

    struct DirtyInstance {
        uint32_t       index;
        InstanceUpdate update;
    };

    struct UpdateState {
        std::vector<DirtyInstance> dirtyInstances{};
        std::vector<uint32_t>      dirtyMaterials{};
        std::vector<uint32_t>      dirtyTextures{};
        bool updateInstances   = false;
        bool updateMaterials   = false;
        bool updateSkybox      = false;
        bool resetAccumulation = false;

        [[nodiscard]] bool any() const {
            return !dirtyInstances.empty() || !dirtyMaterials.empty() || !dirtyTextures.empty()
                || updateInstances || updateMaterials || updateSkybox || resetAccumulation;
        }

        [[nodiscard]] bool  heavy() const {
            return updateInstances || updateMaterials || updateSkybox || !dirtyTextures.empty();
        }

        void clear() { *this = {}; }
    };
}

#endif //COLLECTION_UPDATESTATE_HPP
