//
// Created by igor on 10/17/25.
//

#ifndef COLLECTION_RAY_HPP
#define COLLECTION_RAY_HPP

#include <glm/glm.hpp>

namespace crv::graphics {
    template <typename T>
    struct Ray {
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        Ray() = default;
        Ray(const Vec3& pos, const Vec3& dir, Type tmin, Type tmax):
                pos(pos), dir(dir), tmin(tmin), tmax(tmax) {
        }
        Vec3 pos;
        Vec3 dir;
        Type tmin;
        Type tmax;
    };
}

#endif //COLLECTION_RAY_HPP
