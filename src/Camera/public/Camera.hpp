//
// Created by igor on 7/27/25.
//

#ifndef VULKAN_CAMERA_HPP
#define VULKAN_CAMERA_HPP

#include <stdexcept>
#include <memory>

#include "FlyCamera.hpp"
#include "OrbitalCamera.hpp"

namespace crv::scene {
    AbsCamera* makeCamera(const CameraCreateInfo& createInfo);
    std::unique_ptr<AbsCamera> makeCameraUnique(const CameraCreateInfo& createInfo);
    std::shared_ptr<AbsCamera> makeCameraShared(const CameraCreateInfo& createInfo);
}

#endif //VULKAN_CAMERA_HPP
