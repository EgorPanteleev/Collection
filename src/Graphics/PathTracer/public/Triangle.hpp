//
// Created by igor on 10/19/25.
//

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <glm/glm.hpp>
#include <optional>
#include <tuple>

#include "BBox.hpp"
#include "Ray.hpp"

namespace crv::graphics {
    template<typename T>
    struct Triangle {
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        Triangle() = default;
        Triangle(const Vec3& p0, const Vec3& p1, const Vec3& p2): p0(p0), p1(p1), p2(p2) {}

        BBox<Type> bbox() const {  return { glm::min(glm::min(p0, p1), p2),
                                            glm::max(glm::max(p0, p1), p2)}; }

        Vec3 center() const { return (p0 + p1 + p2) * static_cast<Type>(1./3.); }

        Vec3 p0, p1, p2;
    };
    template<typename T>
    struct PrecomputedTriangle {
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        using Box = BBox<Type>;
        PrecomputedTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2):
        p0(p0), e1(p0 - p1), e2(p2 - p0), N(glm::cross(e1, e2)) {}
        PrecomputedTriangle(const Triangle<Type>& tri): PrecomputedTriangle(tri.p0, tri.p1, tri.p2) {}

        Triangle<Type> castToTriangle() const { return { p0, p0 - e1, e2 + p0 }; }
        Box bbox() const { return castToTriangle().bbox(); }
        Vec3 center() const { return castToTriangle().center(); }
        Vec3 normal() const { return N; }

        std::optional<std::tuple<Type, Type, Type>> intersect(const Ray<Type>& ray, Type eps) const {
            auto c = p0 - ray.pos;
            auto r = cross(ray.dir, c);
            auto inv_det = static_cast<T>(1.) / dot(N, ray.dir);

            auto u = dot(r, e2) * inv_det;
            auto v = dot(r, e1) * inv_det;
            auto w = static_cast<T>(1.) - u - v;

            if (u >= eps && v >= eps && w >= eps) {
                auto t = dot(N, c) * inv_det;
                if (t >= ray.tmin && t <= ray.tmax)
                    return std::make_optional(std::make_tuple(t, u, v));
            }

            return std::nullopt;
        }

        Vec3 p0, e1, e2, N;
    };
}

#endif //TRIANGLE_HPP
