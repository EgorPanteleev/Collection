//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_RENDERSETTINGS_HPP
#define COLLECTION_RENDERSETTINGS_HPP

#include <algorithm>

namespace crv::graphics::vulkan {
    struct RenderSettings {
        int   spp           = 1;
        int   minDepth      = 0;
        int   maxDepth      = 2;
        int   displayMode   = 0;
        bool  nee           = false;
        bool  envNee        = false;
        float aperture      = 0.0f;
        float focusDistance = 10.0f;
        int   renderScale   = 1;
        int   motionScale   = 2;
        float exposure      = 0.8f;
        bool  tonemap       = false;

        [[nodiscard]] uint32_t effectiveRenderScale() const {
            return static_cast<uint32_t>(renderScale);
        }
        [[nodiscard]] uint32_t effectiveMotionScale() const {
            return static_cast<uint32_t>(std::max(motionScale, renderScale));
        }
    };
}

#endif //COLLECTION_RENDERSETTINGS_HPP
