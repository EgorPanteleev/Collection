//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_UBOBUILDER_HPP
#define COLLECTION_UBOBUILDER_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "CoreUtils.hpp"

#include <type_traits>

namespace crv::graphics::vulkan {
    class UBOBuilder {
    public:
        template <typename Type>
        static constexpr std::type_identity<Type> as{};

        explicit UBOBuilder(Context* context): mContext(context) {}

        UBOBuilder& build() { return *this; }

        template <typename Type>
        UBOBuilder& operator()(std::type_identity<Type>, Buffer& buffer) {
            createUBO(mContext->allocator(), static_cast<uint32_t>(sizeof(Type)), buffer);
            return *this;
        }

    private:
        Context* mContext;
    };
}

#endif //COLLECTION_UBOBUILDER_HPP
