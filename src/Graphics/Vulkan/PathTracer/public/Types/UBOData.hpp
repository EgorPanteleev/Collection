//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_UBODATA_HPP
#define COLLECTION_UBODATA_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    using UBOInfo  = std::tuple<uint32_t, Buffer&>;
    struct UBOData: std::vector<UBOInfo> {
        UBOData() = default;
        template <typename Type>
        void add(Buffer& buffer) { emplace_back(sizeof(Type), buffer); }
        void createAll(Context* context) const;
    };

    inline void UBOData::createAll(Context* context) const {
        for (const auto& uboInfo: *this) {
            createUBO(context->allocator(), std::get<0>(uboInfo), std::get<1>(uboInfo));
        }
    }
}

#endif //COLLECTION_UBODATA_HPP