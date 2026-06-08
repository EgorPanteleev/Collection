//
// Created by igor on 6/8/26.
//

#include "Types.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    void SSBOData::createAll(Context* context, QueueFamilyType familyType) const {
        for (const auto& ssboInfo: *this) {
            createSSBO(context->allocator(), std::get<1>(ssboInfo), std::get<2>(ssboInfo));
            copyDataToBuffer(context, familyType, std::get<0>(ssboInfo), std::get<1>(ssboInfo), std::get<2>(ssboInfo));
        }
    }

    void UBOData::createAll(Context* context) const {
        for (const auto& uboInfo: *this) {
            createUBO(context->allocator(), std::get<0>(uboInfo), std::get<1>(uboInfo));
        }
    }
}
