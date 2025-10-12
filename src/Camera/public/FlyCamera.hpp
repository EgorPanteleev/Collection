//
// Created by auser on 5/4/25.
//

#ifndef VULKAN_CAMERA_H
#define VULKAN_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "AbsCamera.hpp"
namespace crv::scene {
    class FlyCamera: public AbsCamera {
    public:
        FlyCamera(const CameraCreateInfo& createInfo);

        void move(float forward_, float right_, float up_) override;
        void rotate(float pitch, float yaw, float roll) override;
        void zoom(float delta) override;
    protected:
    };
}


#endif //VULKAN_CAMERA_H
