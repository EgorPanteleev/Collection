//
// Created by igor on 7/27/25.
//
#include "Camera.hpp"

namespace crv::scene {

    AbsCamera *makeCamera(const CameraCreateInfo &createInfo) {
        switch (createInfo.type) {
            case CameraType::FLY: {
                return new FlyCamera(createInfo);
            }
            case CameraType::ORBITAL: {
                return new OrbitalCamera(createInfo);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported camera type!");
            }
        }
        return nullptr;
    }

    std::unique_ptr<AbsCamera> makeCameraUnique(const CameraCreateInfo &createInfo) {
        switch (createInfo.type) {
            case CameraType::FLY: {
                return std::make_unique<FlyCamera>(createInfo);
            }
            case CameraType::ORBITAL: {
                return std::make_unique<OrbitalCamera>(createInfo);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported camera type!");
            }
        }
        return nullptr;
    }

    std::shared_ptr<AbsCamera> makeCameraShared(const CameraCreateInfo &createInfo) {
        switch (createInfo.type) {
            case CameraType::FLY: {
                return std::make_shared<FlyCamera>(createInfo);
            }
            case CameraType::ORBITAL: {
                return std::make_shared<OrbitalCamera>(createInfo);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported camera type!");
            }
        }
        return nullptr;
    }
}