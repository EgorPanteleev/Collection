//
// Created by igor on 6/16/26.
//

#ifndef COLLECTION_IMGUIFILEDIALOG_HPP
#define COLLECTION_IMGUIFILEDIALOG_HPP

#include <string>
#include <vector>
#include <filesystem>

namespace crv::graphics::vulkan {
    class ImGuiFileDialog {
    public:
        void open(const std::filesystem::path& startDir, std::vector<std::string> exts = {});
        bool draw(const char* title);
        [[nodiscard]] const std::string& result() const { return mResult; }

    private:
        bool drawBreadcrumb();
        void drawPathBar();

        bool                     mVisible       = false;
        std::filesystem::path    mDir{};
        std::string              mSelected{};
        std::string              mResult{};
        std::string              mPathInput{};
        bool                     mPathEditing   = false;
        bool                     mPathFocus     = false;
        bool                     mPathSelectAll = false;
        std::vector<std::string> mFilters{};
    };
}

#endif //COLLECTION_IMGUIFILEDIALOG_HPP
