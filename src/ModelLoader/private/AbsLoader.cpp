//
// Created by auser on 5/5/25.
//

#include "AbsLoader.hpp"

#include <unordered_map>

namespace crv::model {
    AbsLoader::AbsLoader(std::string modelPath) : mModelPath(std::move(modelPath)),
                                                  mVertices(), mIndices() {}

    static glm::vec3 min(const glm::vec3 &v1, const glm::vec3 &v2) {
        return {
                std::min(v1.x, v2.x),
                std::min(v1.y, v2.y),
                std::min(v1.z, v2.z)
        };
    }

    static glm::vec3 max(const glm::vec3 &v1, const glm::vec3 &v2) {
        return {
                std::max(v1.x, v2.x),
                std::max(v1.y, v2.y),
                std::max(v1.z, v2.z)
        };
    }

    void AbsLoader::computeBBox() {
        for (const auto &vert: vertices()) {
            mBBox.min = min(mBBox.min, vert.pos);
            mBBox.max = max(mBBox.max, vert.pos);
        }
    }
}