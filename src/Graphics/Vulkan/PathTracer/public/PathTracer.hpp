//
// Created by igor on 4/8/26.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Context.hpp"

namespace crv::graphics::vulkan {
    class PathTracer {
    public:
        explicit PathTracer(const WindowCreateInfo& windowCreateInfo);
    protected:
        Context mContext{};
    };
}

#endif //COLLECTION_PATHTRACER_HPP