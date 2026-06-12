//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_SSBODATA_HPP
#define COLLECTION_SSBODATA_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    using SSBOInfo = std::tuple<void*, uint32_t, Buffer&>;

    struct SSBOData: std::vector<SSBOInfo> {
        SSBOData() = default;
        template <typename Type>
        void add(const std::vector<Type>& dataBuffer, Buffer& buffer) { emplace_back(const_cast<Type*>(dataBuffer.data()), dataBuffer.size() * sizeof(Type), buffer); }
        void createAll(Context* context, QueueFamilyType familyType) const;
    };

    inline void SSBOData::createAll(Context* context, QueueFamilyType familyType) const {
        for (const auto& ssboInfo: *this) {
            createSSBO(context->allocator(), std::get<1>(ssboInfo), std::get<2>(ssboInfo));
            copyDataToBuffer(context, familyType, std::get<0>(ssboInfo), std::get<1>(ssboInfo), std::get<2>(ssboInfo));
        }
    }
}

#endif //COLLECTION_SSBODATA_HPP