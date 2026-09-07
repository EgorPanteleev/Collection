//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_UBOBUILDER_HPP
#define COLLECTION_UBOBUILDER_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    class UBOBuilder {
    public:
        explicit UBOBuilder(Context* context): mContext(context) {}
        template <typename Type>
        UBOBuilder& add(Buffer& buffer) {
            createUBO(mContext->allocator(), static_cast<uint32_t>(sizeof(Type)), buffer);
            return *this;
        }

    private:
        Context* mContext;
    };
}

#endif //COLLECTION_UBOBUILDER_HPP
