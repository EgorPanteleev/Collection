//
// Created by auser on 5/6/25.
//

#ifndef VULKAN_ASSIMPLOADER_H
#define VULKAN_ASSIMPLOADER_H

#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "AbsLoader.hpp"

namespace crv::model {

    class AssimpLoader : public AbsLoader {
    public:
        AssimpLoader(std::string modelPath);

        bool load(const glm::mat4& model) override;
    private:
        bool loadScene();

        bool loadGeometry();

        bool loadMaterials();

        Node buildNode(const aiNode* node);

        void buildMesh(uint meshIndex);

        template<typename VertexType>
        void processMesh(std::vector<VertexType> &vertices, std::vector<uint32_t> &indices, uint meshIndex);

        template<typename VertexType>
        void optimizeMesh(std::vector<VertexType> &vertices, std::vector<uint32_t> &indices, uint meshIndex);

        void loadTextures(uint materialIndex);

        void loadTexture(Texture::Type textureType, uint materialIndex);

        static bool decodeEmbedded(const aiTexture* aiTex, Texture& texture);

        void loadColors(uint materialIndex);

        const aiScene *mScene;
    };

}
#endif //VULKAN_ASSIMPLOADER_H
