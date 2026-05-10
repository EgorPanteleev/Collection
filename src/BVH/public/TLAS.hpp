//
// Created by igor on 5/9/26.
//

#ifndef COLLECTION_TLAS_HPP
#define COLLECTION_TLAS_HPP

namespace crv::graphics {
    template <typename T>
    struct MeshPrimitive {
        using Mat4 = glm::mat<4, 4, T>;
        using Box = BBox<T>;
        using Vec3 = Box::Vec3;

        MeshPrimitive() = default;
        MeshPrimitive(const Mat4& model, const uint32_t meshIndex, const Box& bbox):
        mModel(model), mMeshIndex(meshIndex) {
            mInvModel = glm::inverse(model);
            mCenter = Vec3(model * glm::vec4(bbox.center(), 1.0));
            Vec3 extent = bbox.size() * static_cast<T>(0.5);
            glm::mat3 absModel = glm::mat3(model);
            absModel[0] = glm::abs(absModel[0]);
            absModel[1] = glm::abs(absModel[1]);
            absModel[2] = glm::abs(absModel[2]);
            Vec3 newExtent = absModel * extent;
            mBbox = { mCenter - newExtent, mCenter + newExtent };
        }

        Box bbox() const { return mBbox; }

        Vec3 center() const { return mCenter; }

        Mat4 mModel;
        Mat4 mInvModel;
        uint32_t mMeshIndex{};
        Box mBbox;
        Vec3 mCenter;
    };

}

#endif //COLLECTION_TLAS_HPP