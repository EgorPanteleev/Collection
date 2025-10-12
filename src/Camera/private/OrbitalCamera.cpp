//
// Created by igor on 7/27/25.
//

#include "OrbitalCamera.hpp"

namespace crv::scene {

    OrbitalCamera::OrbitalCamera(const CameraCreateInfo &createInfo) : AbsCamera(createInfo),
                                                                       mTarget(createInfo.target),
                                                                       mRadius(glm::length(mTarget - mPosition)) {
    }

    void OrbitalCamera::rotate(float pitch, float yaw, float roll) {
        glm::quat pitchQuat = glm::angleAxis(glm::radians(pitch), right());
        glm::quat yawQuat = glm::angleAxis(glm::radians(yaw), mUp);
        glm::quat rollQuat = glm::angleAxis(glm::radians(roll), forward());
        mOrientation = glm::normalize(rollQuat * pitchQuat * yawQuat * mOrientation);
        mPosition = mTarget - forward() * mRadius;
        calculateView();
    }

    void OrbitalCamera::zoom(float delta) {
        mRadius = std::max(0.1f, mRadius - delta);
        mPosition = mTarget - forward() * mRadius;
        calculateView();
    }
}