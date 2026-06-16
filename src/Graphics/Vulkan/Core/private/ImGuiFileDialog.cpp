//
// Created by igor on 6/16/26.
//

#include "ImGuiFileDialog.hpp"

#include <imgui.h>
#include "IconsFontAwesome6.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <system_error>

namespace crv::graphics::vulkan {
    static std::string toLower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static bool extensionAllowed(const std::filesystem::path& path, const std::vector<std::string>& filters) {
        if (filters.empty()) return true;
        std::string ext = toLower(path.extension().string());
        for (const std::string& allowed : filters)
            if (ext == toLower(allowed)) return true;
        return false;
    }

    static bool isImageFile(const std::filesystem::path& path) {
        static const std::vector<std::string> images = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr",
            ".exr", ".ktx", ".dds", ".gif", ".tif", ".tiff"
        };
        std::string ext = toLower(path.extension().string());
        for (const std::string& image : images)
            if (ext == image) return true;
        return false;
    }

    static bool entryRow(const std::string& id, const char* icon, const ImVec4& iconColor,
                         const std::string& name, bool selected, bool allowDouble, bool& doubleClicked) {
        ImGuiSelectableFlags flags = ImGuiSelectableFlags_NoAutoClosePopups;
        if (allowDouble) flags |= ImGuiSelectableFlags_AllowDoubleClick;

        ImVec2 origin  = ImGui::GetCursorPos();
        bool   clicked = ImGui::Selectable(("##" + id).c_str(), selected, flags, ImVec2(0.0f, ImGui::GetTextLineHeight() + 6.0f));
        doubleClicked  = clicked && allowDouble && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

        ImGui::SetCursorPos(ImVec2(origin.x + 6.0f, origin.y + 3.0f));
        ImGui::TextColored(iconColor, "%s", icon);
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::TextUnformatted(name.c_str());
        return clicked;
    }

    static int pathSelectAllCallback(ImGuiInputTextCallbackData* data) {
        bool* pending = static_cast<bool*>(data->UserData);
        if (pending && *pending) {
            data->SelectionStart = 0;
            data->SelectionEnd   = data->BufTextLen;
            data->CursorPos      = data->BufTextLen;
            *pending = false;
        }
        return 0;
    }

    void ImGuiFileDialog::open(const std::filesystem::path& startDir, std::vector<std::string> exts) {
        mDir           = startDir;
        mFilters       = std::move(exts);
        mVisible       = true;
        mPathEditing   = false;
        mPathFocus     = false;
        mPathSelectAll = false;
        mSelected.clear();
        mResult.clear();
        mPathInput.clear();
    }

    bool ImGuiFileDialog::drawBreadcrumb() {
        ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 1.0f, 1.0f, 0.08f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(1.0f, 1.0f, 1.0f, 0.14f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));

        bool hovered = false;
        std::filesystem::path accum;
        bool first = true;
        for (auto it = mDir.begin(); it != mDir.end(); ++it) {
            std::string part = it->string();
            if (part.empty()) continue;
            accum /= *it;
            if (!first) {
                ImGui::SameLine(0.0f, 3.0f);
                ImGui::TextDisabled("/");
                ImGui::SameLine(0.0f, 3.0f);
            }
            std::string label = (part == "/" ? std::string(ICON_FA_HARD_DRIVE) : part) + "##crumb" + accum.string();
            if (ImGui::SmallButton(label.c_str())) {
                mDir = accum;
                mSelected.clear();
            }
            if (ImGui::IsItemHovered()) hovered = true;
            first = false;
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);
        return hovered;
    }

    void ImGuiFileDialog::drawPathBar() {
        if (mPathEditing) {
            char buffer[1024];
            std::snprintf(buffer, sizeof(buffer), "%s", mPathInput.c_str());

            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.13f, 0.13f, 0.13f, 1.0f));
            ImGui::SetNextItemWidth(-1.0f);
            if (mPathFocus)
                ImGui::SetKeyboardFocusHere();
            bool entered = ImGui::InputTextWithHint("##path", "Type or paste a path...", buffer,
                                                    sizeof(buffer),
                                                    ImGuiInputTextFlags_EnterReturnsTrue
                                                    | ImGuiInputTextFlags_CallbackAlways,
                                                    pathSelectAllCallback, &mPathSelectAll);
            ImGui::PopStyleColor();
            if (ImGui::IsItemActive())
                mPathFocus = false;
            mPathInput = buffer;

            if (entered) {
                std::error_code ec;
                std::filesystem::path path(mPathInput);
                if (std::filesystem::is_directory(path, ec)) {
                    mDir = path;
                    mSelected.clear();
                } else if (std::filesystem::is_regular_file(path, ec)) {
                    mSelected = path.string();
                    mDir      = path.parent_path();
                }
            }
            if (entered || ImGui::IsItemDeactivated())
                mPathEditing = false;
            return;
        }

        ImGui::AlignTextToFramePadding();
        ImGui::BeginGroup();
        bool segmentHovered = drawBreadcrumb();
        ImGui::EndGroup();

        ImVec2 mn = ImGui::GetItemRectMin();
        ImVec2 mx = ImGui::GetItemRectMax();
        mx.x = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        if (!segmentHovered && ImGui::IsMouseHoveringRect(mn, mx) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            mPathInput     = mDir.string();
            mPathEditing   = true;
            mPathFocus     = true;
            mPathSelectAll = true;
        }
    }

    bool ImGuiFileDialog::draw(const char* title) {
        if (!mVisible) return false;
        if (!ImGui::IsPopupOpen(title)) ImGui::OpenPopup(title);

        const ImVec4 folderColor{0.86f, 0.71f, 0.36f, 1.0f};
        const ImVec4 imageColor {0.56f, 0.90f, 0.69f, 1.0f};
        const ImVec4 fileColor  {0.62f, 0.62f, 0.62f, 1.0f};

        bool picked = false;
        ImGui::SetNextWindowSize(ImVec2(620, 460), ImGuiCond_FirstUseEver);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 12.0f));
        if (ImGui::BeginPopupModal(title, &mVisible, ImGuiWindowFlags_NoSavedSettings)) {
            drawPathBar();
            ImGui::Spacing();

            const float footer = ImGui::GetTextLineHeightWithSpacing()
                               + ImGui::GetFrameHeightWithSpacing()
                               + ImGui::GetStyle().ItemSpacing.y * 2.0f + 4.0f;

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.065f, 0.065f, 0.065f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
            ImGui::BeginChild("##entries", ImVec2(0.0f, -footer), ImGuiChildFlags_Borders);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 2.0f));

            bool dummy = false;
            if (mDir.has_parent_path() && mDir != mDir.root_path()) {
                if (entryRow("..", ICON_FA_ARROW_UP_LONG, fileColor, "..", false, true, dummy) && dummy) {
                    mDir = mDir.parent_path();
                    mSelected.clear();
                }
            }

            std::vector<std::filesystem::path> dirs, files;
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(mDir, ec)) {
                if (entry.path().filename().string().rfind('.', 0) == 0) continue;
                if (entry.is_directory(ec)) dirs.push_back(entry.path());
                else if (entry.is_regular_file(ec) && extensionAllowed(entry.path(), mFilters))
                    files.push_back(entry.path());
            }
            auto byName = [](const std::filesystem::path& a, const std::filesystem::path& b) {
                return toLower(a.filename().string()) < toLower(b.filename().string());
            };
            std::sort(dirs.begin(), dirs.end(), byName);
            std::sort(files.begin(), files.end(), byName);

            for (const auto& path : dirs) {
                bool doubled = false;
                if (entryRow(path.string(), ICON_FA_FOLDER, folderColor, path.filename().string(), false, true, doubled) && doubled) {
                    mDir = path;
                    mSelected.clear();
                }
            }
            for (const auto& path : files) {
                std::string full      = path.string();
                bool        image     = isImageFile(path);
                bool        doubled   = false;
                if (entryRow(full, image ? ICON_FA_FILE_IMAGE : ICON_FA_FILE, image ? imageColor : fileColor,
                             path.filename().string(), mSelected == full, true, doubled)) {
                    mSelected = full;
                    if (doubled) {
                        mResult  = full;
                        mVisible = false;
                        picked = true;
                        ImGui::CloseCurrentPopup();
                    }
                }
            }

            if (dirs.empty() && files.empty())
                ImGui::TextDisabled("  Empty folder");

            ImGui::PopStyleVar();
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();

            ImGui::Spacing();
            if (mSelected.empty()) {
                ImGui::TextDisabled(ICON_FA_FILE "  No file selected");
            } else {
                ImGui::TextColored(imageColor, ICON_FA_FILE_IMAGE);
                ImGui::SameLine(0.0f, 8.0f);
                ImGui::TextUnformatted(std::filesystem::path(mSelected).filename().string().c_str());
            }
            ImGui::Separator();

            const float btnWidth = 96.0f;
            const float avail     = ImGui::GetContentRegionAvail().x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - (btnWidth * 2.0f + ImGui::GetStyle().ItemSpacing.x));

            const bool canOpen = !mSelected.empty();
            ImGui::BeginDisabled(!canOpen);
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f, 0.42f, 0.30f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.52f, 0.38f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.18f, 0.38f, 0.28f, 1.0f));
            if (ImGui::Button(ICON_FA_CHECK " Open", ImVec2(btnWidth, 0.0f))) {
                mResult  = mSelected;
                mVisible = false;
                picked = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor(3);
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button(ICON_FA_XMARK " Cancel", ImVec2(btnWidth, 0.0f))) {
                mVisible = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
        return picked;
    }
}
