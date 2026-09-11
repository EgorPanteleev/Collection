//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_ENVIRONMENTMAP_HPP
#define COLLECTION_ENVIRONMENTMAP_HPP

#include "Context.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"

namespace crv::graphics::vulkan {
    class EnvironmentMap {
    public:
        EnvironmentMap() = default;
        explicit EnvironmentMap(Context* context) : mContext(context) {}
        void build(const cm::Texture& skybox);
        void disable();

        [[nodiscard]] float    integral()       const { return mIntegral; }
        [[nodiscard]] uint64_t marginalCdfAddr() const { return mMarginalCdfAddr; }
        [[nodiscard]] uint64_t condCdfAddr()     const { return mCondCdfAddr; }
        [[nodiscard]] uint64_t condFuncAddr()    const { return mCondFuncAddr; }
    private:
        Context* mContext = nullptr;
        Buffer   mMarginalCdfBuffer = CRV_NULL_HANDLE;
        Buffer   mCondCdfBuffer     = CRV_NULL_HANDLE;
        Buffer   mCondFuncBuffer    = CRV_NULL_HANDLE;
        uint64_t mMarginalCdfAddr   = 0;
        uint64_t mCondCdfAddr       = 0;
        uint64_t mCondFuncAddr      = 0;
        float    mIntegral          = 0.0f;
    };
}

#endif //COLLECTION_ENVIRONMENTMAP_HPP
