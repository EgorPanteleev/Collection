//
// Created by igor on 10/17/25.
//

#ifndef COLLECTION_RAY_HPP
#define COLLECTION_RAY_HPP

#include <glm/glm.hpp>

#include "Utils.hpp"

namespace crv::graphics {
    template <typename T>
    struct Ray {
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        Ray() = default;
        Ray(const Vec3& pos, const Vec3& dir, Type tmin, Type tmax):
                pos(pos), dir(dir), invDir(static_cast<Type>(1) / dir), tmin(tmin), tmax(tmax) {
            invDirPad.x = utils::addUlp(invDir.x, 2);
            invDirPad.y = utils::addUlp(invDir.y, 2);
            invDirPad.z = utils::addUlp(invDir.z, 2);
        }
        Vec3 pos;
        Vec3 dir;
        Vec3 invDir;
        Vec3 invDirPad;
        Type tmin;
        Type tmax;
    };
}

#endif //COLLECTION_RAY_HPP
