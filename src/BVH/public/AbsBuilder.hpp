//
// Created by igor on 11/1/25.
//

#ifndef ABSBUILDER_HPP
#define ABSBUILDER_HPP

#include <span>

#include "BBox.hpp"
#include "BVH.hpp"
#include "BinnedSAHBuilder.hpp"

namespace crv::graphics {
    template <typename Node>
    class AbsBuilder {
    public:
        using BoxType = BBox<typename Node::Type>;
        using Vec3 = typename BoxType::Vec3;

        enum BuildType {
            BINNED_SAH,
            MINI_TREE
        };

        explicit AbsBuilder(BuildType type): mType(type) {}

        template<typename Primitive>
        BVH<Node> build(std::span<Primitive> prims) const {
            std::vector<BoxType> bboxes;
            bboxes.reserve(prims.size());
            std::vector<Vec3> centers;
            centers.reserve(prims.size());
            for (size_t i = 0; i < prims.size(); ++i) {
                bboxes.push_back(prims[i].bbox());
                centers.push_back(prims[i].center());
            }

            switch (mType) {
                case BINNED_SAH:
                    return BinnedSAHBuilder<Node>(bboxes, centers).build();
                case MINI_TREE:
                    static_assert("Mini tree doesnt implement for now!");
                    break;
            }
        }
    private:
        BuildType mType;
    };
}

#endif //ABSBUILDER_HPP
