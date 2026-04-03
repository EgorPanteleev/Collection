//
// Created by igor on 3/24/26.
//

#ifndef COLLECTION_MATERIAL_HPP
#define COLLECTION_MATERIAL_HPP

#include <random>
#include <glm/glm.hpp>

namespace crv::graphics {

    namespace {
        template <typename Type>
        Type rand01() {
            thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<Type> dist(static_cast<Type>(0), static_cast<Type>(1));
            return dist(rng);
        }
    }

    /*
     * vec3 brdf(normal, wi, wo)
     * wi - light -> point
     * wo - point -> camera
     * return color
     *
     * Type pdf(normal, wi, wo)
     * return probability
     *
     *
    */
    enum class MaterialType {
        LAMBERTIAN,
        UNKNOWN
    };

    template <typename T>
    class Material {
    public:
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        Material(): mType(MaterialType::UNKNOWN) {}
        explicit Material(const MaterialType type): mType(type) {}
        Vec3 sample(const Vec3& N) {
            const Type u = rand01<Type>();
            const Type v = rand01<Type>();
            const Type r = std::sqrt(u);
            Type azimuth = v * 2 * M_PI;
            return { r * std::cos(azimuth), r * std::sin(azimuth), std::sqrt(1 - u) };
        }

        void scatter(const Vec3& N, const Vec3& wo, Vec3& wi, Vec3& brdf, Type& pdf) {
            Vec3 tangent, bitangent;
            coordinateSystem(N, tangent, bitangent);
            Vec3 localSample = sample(N);
            wi = localSample.x * tangent + localSample.y * bitangent + localSample.z * N;

            pdf = PDF(N, wi, wo);
            brdf = BRDF(N, wi, wo);
        }
    protected:
        void coordinateSystem(const Vec3& N, Vec3& tangent, Vec3& bitangent) {
            if (std::abs(N.x) > std::abs(N.z))
                tangent = glm::normalize(Vec3(-N.y, N.x, 0));
            else
                tangent = glm::normalize(Vec3(0, -N.z, N.y));
            bitangent = glm::cross(N, tangent);
        }

        virtual Vec3 BRDF(const Vec3& N, const Vec3& wi, const Vec3& wo) = 0;
        virtual Type PDF(const Vec3& N, const Vec3& wi, const Vec3& wo) = 0;

        MaterialType mType;
    };

    template <typename T>
    class Lambertian: public Material<T> {
    public:
        using Type = T;
        using Vec3 = Material<Type>::Vec3;
        Lambertian(): Material<Type>(MaterialType::LAMBERTIAN) {}
    protected:
        Vec3 BRDF(const Vec3& N, const Vec3& wi, const Vec3& wo) override {
            return Vec3(M_1_PI);
        }
        Type PDF(const Vec3& N, const Vec3& wi, const Vec3& wo) override {
            Type theta = std::max(glm::dot(N, wi), static_cast<Type>(0));
            return theta * M_1_PI;
        }
    };

}

#endif //COLLECTION_MATERIAL_HPP
