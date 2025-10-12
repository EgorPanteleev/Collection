//
// Created by igor on 10/6/25.
//

#include <unordered_map>

#include "Loader.hpp"
#include "AssimpLoader.hpp"

namespace crv::model {
    Loader::Loader(std::string modelPath): mLoader(getLoader(std::move(modelPath))) {
    }

    bool Loader::load() {
        return mLoader->load();
    }

    BBox Loader::bbox() const { return mLoader->bbox(); }
    [[nodiscard]] const std::vector<Mesh>& Loader::meshes() const { return mLoader->meshes(); }
    [[nodiscard]] const std::vector<Material>& Loader::materials() const { return mLoader->materials(); }
    [[nodiscard]] const std::vector<Vertex>& Loader::vertices() const { return mLoader->vertices(); }
    [[nodiscard]] const std::vector<uint32_t>& Loader::indices() const { return mLoader->indices(); }

    std::unique_ptr<AbsLoader> Loader::getLoader(std::string path) {
         /// now only use assimp, in future can be other loaders, like tiny gltf or my own for some file types
         return std::make_unique<AssimpLoader>(std::move(path));
    }
}