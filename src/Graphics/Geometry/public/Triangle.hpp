//
// Created by igor on 10/19/25.
//

#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <glm/glm.hpp>
#include <optional>
#include <tuple>
#include <immintrin.h>

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
        using Vec2 = glm::vec<2, Type>;
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

    struct PreTri16 {
        using Type = float;
        using Vec3 = glm::vec<3, Type>;
        using Vec2 = glm::vec<2, Type>;
        using Box = BBox<Type>;
        // PrecomputedTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2):
        // p0(p0), e1(p0 - p1), e2(p2 - p0), N(glm::cross(e1, e2)) {}
        // PrecomputedTriangle(const Triangle<Type>& tri): PrecomputedTriangle(tri.p0, tri.p1, tri.p2) {}
        //
        // Triangle<Type> castToTriangle() const { return { p0, p0 - e1, e2 + p0 }; }
        // Box bbox() const { return castToTriangle().bbox(); }
        // Vec3 center() const { return castToTriangle().center(); }
        // Vec3 normal() const { return N; }

        std::optional<std::tuple<Type, Type, Type, int>> intersect(const Ray<float>& ray, float eps = 1e-6f) {
            // 1. c = p0 - ray.pos
            __m512 c_x = _mm512_sub_ps(p0x, _mm512_set1_ps(ray.pos.x));
            __m512 c_y = _mm512_sub_ps(p0y, _mm512_set1_ps(ray.pos.y));
            __m512 c_z = _mm512_sub_ps(p0z, _mm512_set1_ps(ray.pos.z));

            // 2. r = cross(ray.dir, c)
            __m512 r_x = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(ray.dir.y), c_z),
                                       _mm512_mul_ps(_mm512_set1_ps(ray.dir.z), c_y));
            __m512 r_y = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(ray.dir.z), c_x),
                                       _mm512_mul_ps(_mm512_set1_ps(ray.dir.x), c_z));
            __m512 r_z = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(ray.dir.x), c_y),
                                       _mm512_mul_ps(_mm512_set1_ps(ray.dir.y), c_x));

            // 3. inv_det = 1 / dot(N, ray.dir)
            __m512 dotNr = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Nx, _mm512_set1_ps(ray.dir.x)),
                                                       _mm512_mul_ps(Ny, _mm512_set1_ps(ray.dir.y))),
                                         _mm512_mul_ps(Nz, _mm512_set1_ps(ray.dir.z)));
            __m512 inv_det = _mm512_div_ps(_mm512_set1_ps(1.0f), dotNr);

            // 4. u = dot(r, e2) * inv_det
            __m512 u = _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r_x, e2x),
                                                                 _mm512_mul_ps(r_y, e2y)),
                                                   _mm512_mul_ps(r_z, e2z)),
                                     inv_det);

            // 5. v = dot(r, e1) * inv_det
            __m512 v = _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r_x, e1x),
                                                                 _mm512_mul_ps(r_y, e1y)),
                                                   _mm512_mul_ps(r_z, e1z)),
                                     inv_det);

            // 6. w = 1 - u - v
            __m512 w = _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(u, v));

            // 7. t = dot(N, c) * inv_det
            __m512 t = _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Nx, c_x),
                                                                 _mm512_mul_ps(Ny, c_y)),
                                                   _mm512_mul_ps(Nz, c_z)),
                                     inv_det);

            // 8. Маска для валидных пересечений
            __mmask16 mask = _mm512_cmp_ps_mask(u, _mm512_set1_ps(eps), _CMP_GE_OQ);
            mask &= _mm512_cmp_ps_mask(v, _mm512_set1_ps(eps), _CMP_GE_OQ);
            mask &= _mm512_cmp_ps_mask(w, _mm512_set1_ps(eps), _CMP_GE_OQ);
            mask &= _mm512_cmp_ps_mask(t, _mm512_set1_ps(ray.tmin), _CMP_GE_OQ);
            mask &= _mm512_cmp_ps_mask(t, _mm512_set1_ps(ray.tmax), _CMP_LE_OQ);

            if (mask == 0) return std::nullopt;

            // 9. Находим минимальный t через AVX-512 reduce
            float tMin = _mm512_mask_reduce_min_ps(mask, t);

            // 10. Определяем индекс минимального t
            int index = -1;
            for (int i = 0; i < 16; ++i) {
                if ((mask >> i) & 1) {
                    float t_val = ((float*)&t)[i]; // safe для маленького количества
                    if (t_val == tMin) {
                        index = i;
                        break;
                    }
                }
            }

            // 11. Извлекаем barycentric coordinates u, v
            float u_hit = ((float*)&u)[index];
            float v_hit = ((float*)&v)[index];

            return std::make_optional(std::make_tuple(tMin, u_hit, v_hit, index));
        }

    public:
        __m512 p0x, p0y, p0z;
        __m512 e1x, e1y, e1z;
        __m512 e2x, e2y, e2z;
        __m512 Nx, Ny, Nz;
    };
}

#endif //TRIANGLE_HPP
