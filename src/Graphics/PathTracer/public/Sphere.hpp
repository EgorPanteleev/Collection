//
// Created by igor on 10/17/25.
//

#ifndef COLLECTION_SPHERE_HPP
#define COLLECTION_SPHERE_HPP

#include "Ray.hpp"
#include "BBox.hpp"

#include <optional>

namespace crv::graphics {
    template <typename T>
    struct Sphere {
        using Type = T;
        using Vec3 = glm::vec<3, Type, glm::defaultp>;
        Sphere() = default;
        Sphere(const Vec3& origin, Type radius): origin(origin), radius(radius) {}

        BBox<Type> bbox() const { return {origin - radius, origin + radius}; }

        template <bool RayNormalized = false>
        std::optional<std::pair<Type, Type>> intersect(const Ray<Type>& ray) const {
            auto oc = ray.pos - origin;
            auto a = RayNormalized ? static_cast<Type>(1.) : dot(ray.dir, ray.dir);
            auto b = static_cast<T>(2.) * dot(ray.dir, oc);
            auto c = dot(oc, oc) - radius * radius;

            auto delta = b * b - static_cast<T>(4.) * a * c;
            if (delta >= 0) {
                auto inv = -static_cast<T>(0.5) / a;
                auto sqrt_delta = std::sqrt(delta);
                auto t0 = glm::max((b + sqrt_delta) * inv, ray.tmin);
                auto t1 = glm::min((b - sqrt_delta) * inv, ray.tmax);
                if (t0 <= t1)
                    return std::make_optional(std::make_pair(t0, t1));
            }

            return std::nullopt;
        }

        Vec3 origin;
        Type radius;
    };
}

#endif //COLLECTION_SPHERE_HPP
