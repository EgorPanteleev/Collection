//
// Created by auser on 5/5/25.
//

#ifndef VULKAN_ABSLOADER_H
#define VULKAN_ABSLOADER_H

#include "Vertex.hpp"
#include "Mesh.hpp"
#include "Material.hpp"
#include "BBox.hpp"

#include <vector>
#include <string>

namespace crv::model {
    class AbsLoader {
    public:
        AbsLoader(std::string modelPath);
        virtual ~AbsLoader() = default;

        [[nodiscard]] BBox bbox() const { return mBBox; }
        [[nodiscard]] const std::vector<Mesh>& meshes() const { return mMeshes; }
        [[nodiscard]] const std::vector<Material>& materials() const { return mMaterials; }
        [[nodiscard]] const std::vector<Vertex>& vertices() const { return mVertices; }
        [[nodiscard]] const std::vector<uint32_t>& indices() const { return mIndices; }

        virtual bool load() = 0;
    protected:
        void computeBBox();
        std::string mModelPath;
        std::vector<Mesh> mMeshes;
        std::vector<Material> mMaterials;
        std::vector<Vertex> mVertices;
        std::vector<uint32_t> mIndices;
        BBox mBBox;
    };
}


#endif //VULKAN_ABSLOADER_H
