//
// Created by auser on 5/6/25.
//

#include "AssimpLoader.hpp"
#include "Message.hpp"

#include <stdexcept>

#include <assimp/postprocess.h>
#include <assimp/config.h>
#include <meshoptimizer.h>
#include <filesystem>

#include <gli/gli.hpp>
#include <stb_image.h>

namespace {
    uint countValidFaces(const aiMesh *Mesh) {
        uint NumValidFaces = 0;

        for (uint i = 0; i < Mesh->mNumFaces; i++) {
            if (Mesh->mFaces[i].mNumIndices == 3) {
                NumValidFaces++;
            }
        }

        return NumValidFaces;
    }

    aiMatrix4x4 glmToAssimp(const glm::mat4& m) {
        aiMatrix4x4 ai;
        ai.a1 = m[0][0]; ai.a2 = m[1][0]; ai.a3 = m[2][0]; ai.a4 = m[3][0];
        ai.b1 = m[0][1]; ai.b2 = m[1][1]; ai.b3 = m[2][1]; ai.b4 = m[3][1];
        ai.c1 = m[0][2]; ai.c2 = m[1][2]; ai.c3 = m[2][2]; ai.c4 = m[3][2];
        ai.d1 = m[0][3]; ai.d2 = m[1][3]; ai.d3 = m[2][3]; ai.d4 = m[3][3];
        return ai;
    }

    glm::mat4 assimpToGlm(const aiMatrix4x4& m) {
        return {
            m.a1, m.b1, m.c1, m.d1,
            m.a2, m.b2, m.c2, m.d2,
            m.a3, m.b3, m.c3, m.d3,
            m.a4, m.b4, m.c4, m.d4
        };
    }
}

namespace crv::model {

    namespace fs = std::filesystem;

#define ASSIMP_LOAD_FLAGS (aiProcess_JoinIdenticalVertices    | \
                           aiProcess_Triangulate              | \
                           aiProcess_GenSmoothNormals         | \
                           aiProcess_LimitBoneWeights         | \
                           aiProcess_ImproveCacheLocality     | \
                           aiProcess_FindDegenerates          | \
                           aiProcess_FindInvalidData          | \
                           aiProcess_GenUVCoords              | \
                           aiProcess_CalcTangentSpace         | \
                           aiProcess_SortByPType              | \
                           aiProcess_FlipUVs)

//#define ASSIMP_LOAD_FLAGS (aiProcess_Triangulate | \
//                           aiProcess_FlipUVs     | \
//                           aiProcess_GenNormals    )


    AssimpLoader::AssimpLoader(std::string modelPath) : AbsLoader(std::move(modelPath)) {}

    bool AssimpLoader::load(const glm::mat4& model) {
        clear();
        Assimp::Importer importer;
        importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
        mScene = importer.ReadFile(mModelPath, ASSIMP_LOAD_FLAGS);
        if (!mScene) {
            ERROR << "Assimp Error: " << importer.GetErrorString();
            return false;
        }
        const aiMatrix4x4 aiModel = glmToAssimp(model);
        mScene->mRootNode->mTransformation = mScene->mRootNode->mTransformation * aiModel;
        return loadScene();
    }

    bool AssimpLoader::loadScene() {
        return loadGeometry() &&
               loadMaterials();
    }

    bool AssimpLoader::loadGeometry() {
        mMeshes.resize(mScene->mNumMeshes);
        for (uint i = 0; i < mScene->mNumMeshes; ++i) buildMesh(i);
        mRoot = buildNode(mScene->mRootNode);
        computeBBox();
        INFO << "Scene size:";
        INFO << "min - (" << mBBox.min.x << ", " << mBBox.min.y << ", " << mBBox.min.z << ")";
        INFO << "max - (" << mBBox.max.x << ", " << mBBox.max.y << ", " << mBBox.max.z << ")";
        return true;
    }

    bool AssimpLoader::loadMaterials() {
        mMaterials.resize(mScene->mNumMaterials);
        for (uint i = 0; i < mScene->mNumMaterials; ++i) {
            loadTextures(i);
            loadColors(i);
        }
        return true;
    }

    void AssimpLoader::buildMesh(const uint meshIndex) {
        Mesh& mesh = mMeshes[meshIndex];
        const aiMesh* aiMeshPtr = mScene->mMeshes[meshIndex];
        mesh.materialIndex = static_cast<int>(aiMeshPtr->mMaterialIndex);
        mesh.validFaces    = countValidFaces(aiMeshPtr);
        mesh.numIndices    = mesh.validFaces * 3;
        mesh.numVertices   = aiMeshPtr->mNumVertices;
        mesh.name          = aiMeshPtr->mName.C_Str();

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        processMesh<Vertex>(vertices, indices, meshIndex);
        mesh.baseVertex = mVertices.size();
        mesh.baseIndex  = mIndices.size();
        mVertices.insert(mVertices.end(), vertices.begin(), vertices.end());
        mIndices.insert(mIndices.end(), indices.begin(), indices.end());
    }

    Node AssimpLoader::buildNode(const aiNode* node) {
        Node result;
        result.name = node->mName.C_Str();
        result.transform = assimpToGlm(node->mTransformation);
        result.meshes.reserve(node->mNumMeshes);
        for (unsigned i = 0; i < node->mNumMeshes; ++i)
            result.meshes.push_back(node->mMeshes[i]);
        result.children.reserve(node->mNumChildren);
        for (unsigned i = 0; i < node->mNumChildren; ++i)
            result.children.push_back(buildNode(node->mChildren[i]));
        return result;
    }

    template<typename VertexType>
    void AssimpLoader::processMesh(std::vector<VertexType> &vertices, std::vector<uint32_t> &indices,
        uint meshIndex) {
        VertexType vert{};
        const aiMesh *mesh = mScene->mMeshes[meshIndex];
        for (size_t i = 0; i < mesh->mNumVertices; ++i) {
            const aiVector3D pos = mesh->mVertices[i];
            vert.pos = glm::vec3(pos.x, pos.y, pos.z);

            if (mesh->mNormals) {
                const aiVector3D normal = mesh->mNormals[i];
                vert.normal = glm::vec3(normal.x, normal.y, normal.z);
            } else {
                vert.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            if (mesh->mTangents) {
                const aiVector3D &tangent = mesh->mTangents[i];
                glm::vec3 computedBitangent = glm::cross(vert.normal, glm::vec3(tangent.x, tangent.y, tangent.z));
                auto B = mesh->mBitangents[i];
                float w = (dot(computedBitangent, glm::vec3(B.x, B.y, B.z)) < 0.0f) ? -1.0f : 1.0f;
                vert.tangent = glm::vec4(tangent.x, tangent.y, tangent.z, w);
            } else {
                vert.tangent = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
            }

            if (mesh->HasTextureCoords(0)) {
                const aiVector3D &texCoord = mesh->mTextureCoords[0][i];
                vert.texCoord0 = glm::vec2(texCoord.x, texCoord.y);
            } else {
                vert.texCoord0 = glm::vec2(0.0f);
            }
            if (mesh->HasTextureCoords(1)) {
                const aiVector3D &texCoord = mesh->mTextureCoords[1][i];
                vert.texCoord1 = glm::vec2(texCoord.x, texCoord.y);
            } else {
                vert.texCoord1 = glm::vec2(0.0f);
            }

            if (mesh->mColors[0]) {
                const aiColor4D &color = *mesh->mColors[0];
                vert.color = glm::vec3(color.r, color.g, color.b);
            } else {
                vert.color = glm::vec3(1.0f);
            }

            vertices.push_back(vert);
        }

        for (size_t i = 0; i < mesh->mNumFaces; ++i) {
            const aiFace &face = mesh->mFaces[i];
            if (face.mNumIndices != 3) {
                WARNING << "face " << i << " has " << face.mNumIndices << " indices";
                continue;
            }
            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[1]);
            indices.push_back(face.mIndices[2]);
        }
        optimizeMesh<Vertex>(vertices, indices, meshIndex);
    }

    template<typename VertexType>
    void AssimpLoader::optimizeMesh(std::vector<VertexType> &vertices, std::vector<uint32_t> &indices, uint meshIndex) {
        size_t numIndices = indices.size();
        size_t numVertices = vertices.size();
        if (numIndices == 0 || numVertices == 0) {
            indices.clear();
            vertices.clear();
            mMeshes[meshIndex].numIndices = 0;
            mMeshes[meshIndex].numVertices = 0;
            return;
        }

        std::vector<unsigned int> remap(numIndices);
        size_t optVertexCount = meshopt_generateVertexRemap(remap.data(),    // dst addr
                                                            indices.data(),  // src indices
                                                            numIndices,      // ...and size
                                                            vertices.data(), // src vertices
                                                            numVertices,     // ...and size
                                                            sizeof(VertexType)); // stride
        // Allocate a local index/vertex arrays
        std::vector<uint> optIndices;
        std::vector<VertexType> optVertices;
        optIndices.resize(numIndices);
        optVertices.resize(optVertexCount);
        // Optimization #1: remove duplicate vertices
        meshopt_remapIndexBuffer(optIndices.data(), indices.data(), numIndices, remap.data());
        meshopt_remapVertexBuffer(optVertices.data(), vertices.data(), numVertices, sizeof(VertexType), remap.data());
        // Optimization #2: improve the locality of the vertices
        meshopt_optimizeVertexCache(optIndices.data(), optIndices.data(), numIndices, optVertexCount);
        // Optimization #3: reduce pixel overdraw
        meshopt_optimizeOverdraw(optIndices.data(), optIndices.data(), numIndices, &(optVertices[0].pos.x),
                                 optVertexCount, sizeof(VertexType), 1.05f);
        // Optimization #4: optimize access to the vertex buffer
        meshopt_optimizeVertexFetch(optVertices.data(), optIndices.data(), numIndices, optVertices.data(),
                                    optVertexCount, sizeof(VertexType));
        // Optimization #5: create a simplified version of the model
        float threshold = 1.0f;
        size_t targetIndexCount = (size_t) (numIndices * threshold);

        float targetError = 0.0f;
        std::vector<unsigned int> SimplifiedIndices(optIndices.size());
        size_t optIndexCount = meshopt_simplify(SimplifiedIndices.data(), optIndices.data(), numIndices,
                                                &optVertices[0].pos.x, optVertexCount, sizeof(VertexType),
                                                targetIndexCount, targetError);

        static int num_indices = 0;
        num_indices += (int) numIndices;
        static int opt_indices = 0;
        opt_indices += (int) optIndexCount;
        SimplifiedIndices.resize(optIndexCount);

        // Concatenate the local arrays into the class attributes arrays
        indices = SimplifiedIndices;
        vertices = optVertices;

        mMeshes[meshIndex].numIndices = (uint) optIndexCount;
        mMeshes[meshIndex].numVertices = (uint) optVertexCount;
    }

    void AssimpLoader::loadTextures(uint materialIndex) {
        for (uint texType = 0; texType < Texture::UNKNOWN; ++texType) {
            loadTexture((Texture::Type) texType, materialIndex);
        }
    }

    bool AssimpLoader::decodeEmbedded(const aiTexture* aiTex, Texture& texture) {
        if (aiTex->mHeight == 0) {
            int width = 0, height = 0, channels = 0;
            void* pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(aiTex->pcData),
                                                 static_cast<int>(aiTex->mWidth),
                                                 &width, &height, &channels, STBI_rgb_alpha);
            if (!pixels) return false;
            texture.mDataByLevel.emplace_back(pixels, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            return true;
        }
        const size_t texels = static_cast<size_t>(aiTex->mWidth) * aiTex->mHeight;
        if (texels == 0) return false;
        auto* pixels = static_cast<uint8_t*>(std::malloc(texels * 4));
        if (!pixels) return false;
        for (size_t i = 0; i < texels; ++i) {
            pixels[i * 4 + 0] = aiTex->pcData[i].r;
            pixels[i * 4 + 1] = aiTex->pcData[i].g;
            pixels[i * 4 + 2] = aiTex->pcData[i].b;
            pixels[i * 4 + 3] = aiTex->pcData[i].a;
        }
        texture.mDataByLevel.emplace_back(pixels, aiTex->mWidth, aiTex->mHeight);
        return true;
    }

    void AssimpLoader::loadTexture(Texture::Type textureType, uint materialIndex) {
        const aiMaterial *material = mScene->mMaterials[materialIndex];
        aiTextureType assimpType = Texture::toAssimpType(textureType);
        Texture texture;
        aiString aiPath;
        texture.mFormat = toTextureFormat(textureType);
        texture.mType = textureType;
        if (textureType == Texture::BASE_COLOR && material->GetTextureCount(assimpType) <= 0)
            assimpType = aiTextureType_DIFFUSE;
        const unsigned int assimpIndex = textureType == Texture::CLEARCOAT_ROUGHNESS ? 1u : 0u;
        if (material->GetTextureCount(assimpType) <= assimpIndex or
            material->GetTexture(assimpType, assimpIndex, &aiPath, NULL, NULL, NULL, NULL, NULL) != AI_SUCCESS) {
            return;
        }
        std::string path = aiPath.C_Str();
        std::replace(path.begin(), path.end(), '\\', '/');
        const aiTexture *aiTex = mScene->GetEmbeddedTexture(path.c_str());
        if (aiTex) {
            if (!decodeEmbedded(aiTex, texture)) {
                WARNING << "Failed to decode embedded texture " << path << " in " << mModelPath;
                return;
            }
            texture.mPath = mModelPath + "#" + path;
            texture.mType = textureType;
        } else {
            fs::path modelPath(mModelPath);
            std::string dirPath = modelPath.parent_path();
            path = dirPath + "/" + path;
            texture = AbsLoader::loadTexture(path, textureType);
        }
        texture.mName = fs::path(path).filename().string();
        mMaterials[materialIndex].mTextures[textureType] = texture;
    }

    void AssimpLoader::loadColors(uint materialIndex) {
        Material &material = mMaterials[materialIndex];
        const aiMaterial *aiMat = mScene->mMaterials[materialIndex];

        material.mName = aiMat->GetName().C_Str();

        aiColor4D ambientColor(0);
        if (aiMat->Get(AI_MATKEY_COLOR_AMBIENT, ambientColor) == AI_SUCCESS) {
            material.ambientColor.r = ambientColor.r;
            material.ambientColor.g = ambientColor.g;
            material.ambientColor.b = ambientColor.b;
            material.ambientColor.a = std::min(ambientColor.a, 1.0f);
        }
        aiColor4D emissiveColor(0);
        if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) == AI_SUCCESS) {
            material.ambientColor.r += emissiveColor.r;
            material.ambientColor.g += emissiveColor.g;
            material.ambientColor.b += emissiveColor.b;
            material.ambientColor.a += emissiveColor.a;
            material.ambientColor.a = std::min(material.ambientColor.a, 1.0f);
        }

        aiColor4D diffuseColor(0);
        if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) == AI_SUCCESS) {
            material.diffuseColor.r = diffuseColor.r;
            material.diffuseColor.g = diffuseColor.g;
            material.diffuseColor.b = diffuseColor.b;
            material.diffuseColor.a = std::min(diffuseColor.a, 1.0f);
        }

        aiColor4D baseColor(0);
        if (aiMat->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS) {
            material.diffuseColor.r = baseColor.r;
            material.diffuseColor.g = baseColor.g;
            material.diffuseColor.b = baseColor.b;
            material.diffuseColor.a = std::min(baseColor.a, 1.0f);
        }

        aiColor4D emissive(0);
        if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive) == AI_SUCCESS) {
            material.emissiveColor = glm::vec3(emissive.r, emissive.g, emissive.b);
        }
        aiMat->Get(AI_MATKEY_EMISSIVE_INTENSITY, material.emissiveStrength);
        aiMat->Get(AI_MATKEY_METALLIC_FACTOR, material.metallic);
        aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, material.roughness);
        aiMat->Get(AI_MATKEY_CLEARCOAT_FACTOR, material.clearcoat);
        aiMat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, material.clearcoatRoughness);
        aiMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, material.transmission);
        aiMat->Get(AI_MATKEY_REFRACTI, material.ior);
        aiMat->Get(AI_MATKEY_SPECULAR_FACTOR, material.specularFactor);

        aiColor4D specularColor(0.0f, 0.0f, 0.0f, 0.0f);
        if (aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == AI_SUCCESS) {
            material.specularColor.r = specularColor.r;
            material.specularColor.g = specularColor.g;
            material.specularColor.b = specularColor.b;
            material.specularColor.a = std::min(specularColor.a, 1.0f);
        }

        float opaquenessThreshold = 0.05f;
        float opacity = 1.0f;

        if (aiMat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
            material.mTransparencyFactor = std::clamp(1.0f - opacity, 0.0f, 1.0f);
            if (material.mTransparencyFactor >= 1.0f - opaquenessThreshold) {
                material.mTransparencyFactor = 0.0f;
            }
        }

        aiColor4D TransparentColor;
        if (aiMat->Get(AI_MATKEY_COLOR_TRANSPARENT, TransparentColor) == AI_SUCCESS) {
            float Opacity = std::max(std::max(TransparentColor.r, TransparentColor.g), TransparentColor.b);
            material.mTransparencyFactor = std::clamp(Opacity, 0.0f, 1.0f);
            if (material.mTransparencyFactor >= 1.0f - opaquenessThreshold) {
                material.mTransparencyFactor = 0.0f;
            }

            material.mAlphaTest = 0.5f;
        }
    }
}