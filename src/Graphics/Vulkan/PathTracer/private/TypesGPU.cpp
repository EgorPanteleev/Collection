//
// Created by igor on 5/24/26.
//

#include "TypesGPU.hpp"

namespace crv::graphics::vulkan {
    RasterInstanceGPU::RasterInstanceGPU(const MeshInstance& instance): texIndex(instance.texIndex) {
        model = instance.transform.matrix();
        invModel = glm::inverse(model);
    }

    TracerInstanceGPU::TracerInstanceGPU(const MeshInstance& instance):  baseNode(instance.baseNode),
        baseTri(instance.baseTri), texIndex(instance.texIndex) {
        model = instance.transform.matrix();
        invModel = glm::inverse(model);
    }

}