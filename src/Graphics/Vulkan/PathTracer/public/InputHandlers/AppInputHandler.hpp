//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_APPINPUTHANDLER_HPP
#define COLLECTION_APPINPUTHANDLER_HPP

#include "Command.hpp"
#include "AbsCamera.hpp"

#include <string>
#include <vector>

namespace crv::graphics::vulkan {
    namespace cs = scene;
    class PathTracerApp;

    class AppInputHandler {
    public:
        void apply(const Command& command, PathTracerApp* app) const;
        void applySelection(PathTracerApp* app, uint32_t id, bool additive) const;
    private:
        void setCamera(PathTracerApp* app, cs::CameraType type) const;
        void pick(PathTracerApp* app) const;
        void clearSelection(PathTracerApp* app) const;
        void selectInstance(PathTracerApp* app, uint32_t index, bool additive) const;
        void regionSelect(PathTracerApp* app, int x0, int y0, int x1, int y1, bool additive) const;
        void duplicateInstances(PathTracerApp* app, const std::vector<uint32_t>& indices) const;
        void removeInstances(PathTracerApp* app, const std::vector<uint32_t>& indices) const;
        void addMaterial(PathTracerApp* app, uint32_t instanceIndex) const;
        void uploadTexture(PathTracerApp* app, const std::string& path, uint32_t materialIndex, int textureType) const;
        void loadSkybox(PathTracerApp* app, const std::string& path) const;
        void removeSkybox(PathTracerApp* app) const;
        void updateInstanceTransform(PathTracerApp* app, uint32_t index) const;
        void updateInstance(PathTracerApp* app, uint32_t index) const;
        void updateMaterial(PathTracerApp* app, uint32_t index) const;
        void updateImage(PathTracerApp* app) const;
        void toggleControlPanel(PathTracerApp* app) const;
        void saveImage(PathTracerApp* app) const;
        void saveScene(PathTracerApp* app) const;
    };
}

#endif //COLLECTION_APPINPUTHANDLER_HPP
