//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_SSBOBUILDER_HPP
#define COLLECTION_SSBOBUILDER_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "CoreUtils.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    class SSBOBuilder {
    public:
        SSBOBuilder(Context* context, QueueFamilyType familyType): mContext(context), mFamily(familyType) {}
        template <typename Type>
        SSBOBuilder& add(const std::vector<Type>& data, Buffer& buffer) {
            const auto size = static_cast<uint32_t>(data.size() * sizeof(Type));
            createSSBO(mContext->allocator(), size, buffer);
            copyDataToBuffer(mContext, mFamily, data.data(), size, buffer);
            return *this;
        }

    private:
        Context*        mContext;
        QueueFamilyType mFamily;
    };
}

#endif //COLLECTION_SSBOBUILDER_HPP
