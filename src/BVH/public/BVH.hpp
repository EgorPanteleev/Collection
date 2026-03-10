//
// Created by igor on 3/10/26.
//

#ifndef COLLECTION_BVH_HPP
#define COLLECTION_BVH_HPP

#include <vector>

namespace crv::graphics {
    template <typename Node>
    class SweepSAHBuilder;

    template <typename Node>
    class BVH {
    public:
        friend class SweepSAHBuilder<Node>;


    protected:
        std::vector<Node> mNodes;
    };
}

#endif //COLLECTION_BVH_HPP