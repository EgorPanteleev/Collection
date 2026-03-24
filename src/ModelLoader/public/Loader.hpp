//
// Created by igor on 10/6/25.
//

#ifndef VULKAN_LOADER_HPP
#define VULKAN_LOADER_HPP

#include <string>
#include <vector>
#include <memory>

#include "AbsLoader.hpp"
#include "BBox.hpp"
#include "Mesh.hpp"
#include "Material.hpp"
#include "Vertex.hpp"

namespace crv::model {
    class AbsLoader;

    class Loader {
    public:
        using Box = AbsLoader::Box;
        Loader(): Loader("") {}
        Loader(std::string modelPath);
        virtual ~Loader() = default;

        virtual bool load(const glm::mat4& model = glm::mat4(1.));

        void setModel(const std::string& modelPath) const { mLoader->setModel(modelPath); }
        [[nodiscard]] Box bbox() const;
        [[nodiscard]] const std::vector<Mesh>& meshes() const;
        [[nodiscard]] const std::vector<Material>& materials() const;
        [[nodiscard]] const std::vector<Vertex>& vertices() const;
        [[nodiscard]] const std::vector<uint32_t>& indices() const;
    protected:
        static std::unique_ptr<AbsLoader> getLoader(std::string path);

        std::unique_ptr<AbsLoader> mLoader;
    };

}

#endif //VULKAN_LOADER_HPP
