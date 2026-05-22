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
#include <gli/gli.hpp>

namespace crv::model {
    class AbsLoader {
    public:
        using Box = graphics::BBox<float>;
        AbsLoader() = default;
        AbsLoader(std::string modelPath);
        virtual ~AbsLoader() = default;

        void setModel(const std::string& path) { mModelPath = path; }
        [[nodiscard]] Box bbox() const { return mBBox; }
        [[nodiscard]] const std::vector<Mesh>& meshes() const { return mMeshes; }
        [[nodiscard]] const std::vector<Material>& materials() const { return mMaterials; }
        [[nodiscard]] const std::vector<Vertex>& vertices() const { return mVertices; }
        [[nodiscard]] const std::vector<uint32_t>& indices() const { return mIndices; }
        void clear();
        static Texture::Format toTextureFormat(Texture::Type modelTexType);
        static Texture::Format toTextureFormat(gli::texture::format_type gliFormat);
        [[nodiscard]] static Texture loadTexture(const std::string& path, Texture::Type type);
        [[nodiscard]] static Texture colorTexture(const glm::vec3& color, Texture::Type texType);
        [[nodiscard]] static Texture emptyTexture(Texture::Type texType);
        virtual bool load(const glm::mat4& model) = 0;
    protected:
        void computeBBox();
        std::string mModelPath;
        std::vector<Mesh> mMeshes;
        std::vector<Material> mMaterials;
        std::vector<Vertex> mVertices;
        std::vector<uint32_t> mIndices;
        Box mBBox;
    };
}


#endif //VULKAN_ABSLOADER_H
