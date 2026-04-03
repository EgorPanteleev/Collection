//
// Created by igor on 4/3/26.
//

#ifndef COLLECTION_LIGHT_HPP
#define COLLECTION_LIGHT_HPP

#include <glm/glm.hpp>

namespace crv::graphics {
    enum class LightType {
        DIRECTIONAL,
        POINT,
        AREA,
        UNKNOWN
    };

    template <typename T>
    struct LightSample {
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        Vec3 direction;
        Vec3 radiance;
        Type distance;
    };

    template <typename T>
    class Light {
    public:
        using Type = T;
        using Vec3 = glm::vec<3, Type>;
        using Sample = LightSample<Type>;
        explicit Light(const LightType type, Type intensity): mType(type), mIntensity(intensity) {}
        explicit Light(const LightType type): mType(type) {}
        Light(): mType(LightType::UNKNOWN) {}
        virtual ~Light() = default;
        virtual Sample sample(const Vec3& P) const = 0;
    protected:
        LightType mType;
        Type mIntensity;
    };

    template <typename T>
    class DirectionalLight: public Light<T> {
    public:
        using Type = T;
        using Vec3 = Light<T>::Vec3;
        using Sample = Light<T>::Sample;
        explicit DirectionalLight(Type intensity, const Vec3& dir): Light<Type>(LightType::DIRECTIONAL, intensity) {
            mDir = glm::normalize(dir);
        }
        DirectionalLight(): Light<Type>(LightType::DIRECTIONAL) {}

        Sample sample(const Vec3& P) const override {
            return {
                .direction = mDir,
                .radiance = Vec3( this->mIntensity ),
                .distance = std::numeric_limits<T>::max()
            };
        }
    protected:
        Vec3 mDir;
    };

    template <typename T>
    class PointLight: public Light<T> {
    public:
        using Type = T;
        using Vec3 = Light<T>::Vec3;
        using Sample = Light<T>::Sample;
        explicit PointLight(Type intensity, const Vec3& pos): Light<Type>(LightType::DIRECTIONAL, intensity) {
            mPosition = pos;
        }
        PointLight(): Light<T>(LightType::POINT) {}
        Sample sample(const Vec3& P) const override {
            Type distance = glm::distance(mPosition, P);
            return {
                .direction = glm::normalize(mPosition - P),
                .radiance = Vec3( this->mIntensity / (distance * distance) ),
                .distance = distance
            };
        }
    protected:
        Vec3 mPosition;
    };
}

#endif //COLLECTION_LIGHT_HPP