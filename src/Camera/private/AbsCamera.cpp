//
// Created by igor on 7/27/25.
//

#include "AbsCamera.hpp"

namespace crv::scene {

    AbsCamera::AbsCamera(const CameraCreateInfo &createInfo) :
            mType(createInfo.type), mPosition(createInfo.pos), mUp(createInfo.up),
            mFOV(createInfo.FOV), mAspectRatio(createInfo.aspectRatio),
            mNearPlane(createInfo.nearPlane), mFarPlane(createInfo.farPlane) {
        calculateProjection();
        initOrientation(createInfo.target);
    }

    void AbsCamera::initOrientation(const glm::vec3 &target) {
        glm::vec3 forward_ = glm::normalize(target - mPosition);
        glm::vec3 right_ = glm::normalize(glm::cross(forward_, mUp));
        glm::vec3 camUp_ = glm::cross(right_, forward_);
        glm::mat3 rotationMatrix(right_, camUp_, -forward_);
        mOrientation = glm::quat_cast(rotationMatrix);
        calculateView();
    }

    void AbsCamera::calculateProjection() {
        mProjectionMatrix = glm::perspective(glm::radians(mFOV), mAspectRatio,
                                             mNearPlane, mFarPlane);
        mProjectionMatrix[1][1] *= -1;
    }

    void AbsCamera::calculateView() {
        mViewMatrix = glm::lookAt(mPosition, target(), up());
    }
}