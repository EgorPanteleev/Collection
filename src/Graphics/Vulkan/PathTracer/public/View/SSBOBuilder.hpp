//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_SSBOBUILDER_HPP
#define COLLECTION_SSBOBUILDER_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "CoreUtils.hpp"

#include <algorithm>
#include <vector>

namespace crv::graphics::vulkan {
    class SSBOBuilder {
    public:
        SSBOBuilder(Context* context, QueueFamilyType familyType): mContext(context), mFamily(familyType) {}
        template <typename Type>
        SSBOBuilder& add(const std::vector<Type>& data, Buffer& buffer) {
            const auto capacity = std::max<size_t>(data.size(), 1);
            createSSBO(mContext->allocator(), static_cast<uint32_t>(capacity * sizeof(Type)), buffer);
            copyDataToBuffer(mContext, mFamily, data.data(),
                             static_cast<uint32_t>(data.size() * sizeof(Type)), buffer);
            return *this;
        }

    private:
        Context*        mContext;
        QueueFamilyType mFamily;
    };
}

#endif //COLLECTION_SSBOBUILDER_HPP
