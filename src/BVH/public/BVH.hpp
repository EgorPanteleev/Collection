//
// Created by igor on 11/1/25.
//

#ifndef BVH_HPP
#define BVH_HPP

#include <vector>

namespace crv::graphics {
    template <typename Node>
    class BVH {
    public:

        std::vector<Node> mNodes;
        std::vector<size_t> mPrimIds;
    };
}

#endif //BVH_HPP
