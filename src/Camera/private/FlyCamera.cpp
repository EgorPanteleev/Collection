//
// Created by auser on 5/4/25.
//

#include "FlyCamera.hpp"

namespace crv::scene {

    FlyCamera::FlyCamera(const CameraCreateInfo &createInfo) : AbsCamera(createInfo) {
    }

    void FlyCamera::move(float forward_, float right_, float up_) {
        mPosition += forward_ * forward() + right_ * right() + up_ * up();
        calculateView();
    }

    void FlyCamera::rotate(float pitch, float yaw, float roll) {
        glm::quat pitchQuat = glm::angleAxis(glm::radians(pitch), right());
        glm::quat yawQuat = glm::angleAxis(glm::radians(yaw), mUp);
        glm::quat rollQuat = glm::angleAxis(glm::radians(roll), forward());
        mOrientation = glm::normalize(rollQuat * pitchQuat * yawQuat * mOrientation);
        calculateView();
    }

    void FlyCamera::zoom(float delta) {
        mFOV = std::min(135.0f, std::max(1.0f, mFOV - delta));
        calculateProjection();
    }
}