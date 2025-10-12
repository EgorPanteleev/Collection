//
// Created by igor on 7/27/25.
//

#ifndef VULKAN_ORBITALCAMERA_HPP
#define VULKAN_ORBITALCAMERA_HPP

#include "AbsCamera.hpp"
#include <iostream>

namespace crv::scene {
    class OrbitalCamera: public AbsCamera {
    public:
        OrbitalCamera(const CameraCreateInfo& createInfo);

        glm::vec3 target() const override { return mTarget; }

        void move(float forward_, float right_, float up_) override {};
        void rotate(float pitch, float yaw, float roll) override;
        void zoom(float delta) override;
    protected:

        glm::vec3 mTarget;
        float mRadius;
    };
}

#endif //VULKAN_ORBITALCAMERA_HPP
