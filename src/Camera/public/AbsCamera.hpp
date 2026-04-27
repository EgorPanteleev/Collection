//
// Created by igor on 7/27/25.
//

#ifndef VULKAN_ABSCAMERA_H
#define VULKAN_ABSCAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

namespace crv::scene {
    enum class CameraType {
        FLY,
        ORBITAL,
        UNKNOWN
    };

    struct CameraCreateInfo {
        CameraType type = CameraType::UNKNOWN;
        glm::vec3 pos = glm::vec3(0);
        glm::vec3 target = glm::vec3(0, 0, -1);
        glm::vec3 up = glm::vec3(0, 1, 0);
        float zoom = 1;
        float FOV = 60;
        float aspectRatio = 16.0f / 9.0f;
        float nearPlane = 0.0f;
        float farPlane = 1000.0f;
    };

    class AbsCamera {
    public:
        virtual ~AbsCamera() = default;

        using State = std::tuple<glm::vec3, glm::quat, float>;
        AbsCamera() = default;
        AbsCamera(const CameraCreateInfo& createInfo);

        glm::vec3 forward() const { return glm::normalize(mOrientation * glm::vec3(0, 0, -1)); }
        glm::vec3 right() const { return glm::normalize(mOrientation * glm::vec3(1, 0, 0)); }
        glm::vec3 up() const { return glm::normalize(mOrientation * glm::vec3(0, 1, 0)); }
        virtual glm::vec3 target() const { return mPosition + forward(); }
        glm::vec3 position() const { return mPosition; }
        glm::quat orientation() const { return mOrientation; }
        State state() const { return { mPosition, mOrientation, mFOV }; }
        glm::mat4 projectionMatrix() const { return mProjectionMatrix; }
        glm::mat4 viewMatrix() const { return mViewMatrix; }
        CameraType type() const { return mType; }
        float FOV() const { return mFOV; }
        float aspectRatio() const { return mAspectRatio; }
        float nearPlane() const { return mNearPlane; }
        float farPlane() const { return mFarPlane; }

        virtual void move(float forward_, float right_, float up_) = 0;
        virtual void rotate(float pitch, float yaw, float roll) = 0;
        virtual void zoom(float delta) = 0;

        void setPosition(const glm::vec3& position) { mPosition = position; calculateView(); }
        void setOrientation(const glm::quat& orientation) { mOrientation = orientation; calculateView(); }

    protected:
        void calculateProjection();
        void calculateView();
        void initOrientation(const glm::vec3& target);

        CameraType mType = CameraType::UNKNOWN;

        glm::vec3  mPosition{};
        glm::vec3  mUp{};
        glm::quat  mOrientation{};

        glm::mat4  mProjectionMatrix{};
        glm::mat4  mViewMatrix{};

        float mFOV         = 0;
        float mAspectRatio = 0;
        float mNearPlane   = 0;
        float mFarPlane    = 0;
    };
}

#endif //VULKAN_ABSCAMERA_H
