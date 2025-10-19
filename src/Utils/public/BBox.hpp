//
// Created by auser on 6/18/25.
//

#ifndef VULKAN_BBOX_H
#define VULKAN_BBOX_H

#include <glm/glm.hpp>

namespace crv::graphics {
    template <typename T>
    struct BBox {
        using Type = T;
        using Vec3 = glm::vec<3, Type, glm::defaultp>;

        BBox(): min(-std::numeric_limits<Type>::max()), max(std::numeric_limits<Type>::max()) {}
        BBox(const Vec3& min, const Vec3& max): min(min), max(max) {}

        Type width()  const { return max.x - min.x; }
        Type height() const { return max.y - min.y; }
        Type depth()  const { return max.z - min.z; }
        Vec3 size()   const { return { width(), height(), depth() }; }
        BBox& operator+=(const BBox& other) {
            min = glm::min( min, other.min );
            max = glm::max( max, other.max );
            return this;
        }
        Vec3 min;
        Vec3 max;
    };

    template <typename T>
    BBox<T> operator+( BBox<T> left, const BBox<T>& right ) {
        left += right;
        return left;
    }
}

#endif //VULKAN_BBOX_H
