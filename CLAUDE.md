# Collection

A C++23 rendering playground for exploring ray tracing and real-time rendering. It started
as CPU ray tracing and grew into a **real-time GPU path tracer built on Vulkan** (with hardware
ray tracing), plus a hybrid rasterizer/ray-tracer and the original CPU path tracer.

## Active vs legacy

- **Active work lives in `src/Graphics/Vulkan/`.** This is the current focus.
- `src/Old/` is abandoned legacy code — ignore it.
- `src/Graphics/PathTracer/` (non-Vulkan) and `src/Applications/CPUPathTracer/` are the older
  CPU path tracer. The Vulkan **Hybrid** module currently does not compile on this branch
  (references APIs that were since renamed); treat it as behind unless told otherwise.

## Layout

Each module under `src/` follows a `public/` (headers) + `private/` (sources) convention and is a
CMake static library. CMake uses `GLOB_RECURSE`, so **new files/dirs need no CMakeLists edits** —
just place them under `public/` or `private/`.

Supporting modules: `Message` (logging), `Utils`, `ThreadPool`, `BVH`, `Camera`, `ModelLoader`
(assimp/gli-based asset loading), `Graphics/Geometry`.

Third-party (in `thirdparty/`): assimp, gli, imgui, imguizmo, nlohmann/json, meshoptimizer, stb,
VulkanMemoryAllocator.

## Vulkan GPU path tracer — `src/Graphics/Vulkan/`

Three sub-modules:

- **`Core`** (`Vulkan::Core`) — Vulkan wrappers and helpers: `Context`, `Swapchain`, `Buffer`,
  `Image`, descriptor management (`DescriptorManager` + `Wrap/` set/pool/layout wrappers), etc.
- **`PathTracer`** (`Vulkan::PathTracer`) — the main app. Library target `VulkanPathTracer`,
  executable `PathTracerGPU` (from `applications/PathTracer.cpp`, entry `PathTracerApp`).
- **`Hybrid`** (`Vulkan::Hybrid`) — rasterizer + ray-tracing hybrid; currently behind.

### PathTracer architecture (MVC-style separation)

- **Model** — `Model/Scene.hpp` (backend-agnostic scene: meshes, instances, materials, textures,
  lights, skybox) plus `Model/*` value types (`InstanceData`, `Material`, `MeshData`,
  `DirectLight`, `RenderSettings`). Loads/saves JSON scenes.
- **View** — `View/AppUI` (ImGui/ImGuizmo control panel). Reads/edits the `Scene` and emits
  commands; holds no GPU/render state.
- **GPU service** — `ResourceManager` turns the `Scene` into GPU resources (BLAS/TLAS, SSBO/UBO
  buffers, textures, env map). Holds a `Scene*`; no scene getters. `Types/SSBOBuilder` and
  `Types/UBOBuilder` are small fluent helpers for buffer creation/upload.
- **Render passes** — `RayTracerPass` (RT pipeline), `RasterizerPass` (selection/G-buffer),
  `PostprocessPass` (compute tonemap/composite).
- **Orchestration** — `PathTracerApp` owns everything and runs the frame loop. Input flows as
  `Command`s through a `CommandStream`; `InputHandlers/CameraInputHandler` and
  `InputHandlers/AppInputHandler` consume them. `Command.hpp` defines `CommandType` (high 8 bits =
  `CommandTarget` CAMERA/SCENE/VIEW/APP) with `std::variant` payloads.

Shaders are **Slang** in `PathTracer/shaders/`, compiled to SPIR-V by the CMake build (see
`scripts/compile_slang_helper.sh`).

## Build & run

Configured build dirs exist (`build/`, `debug/`, `release/`). Build the main target:

```sh
cmake --build build -j 4 --target PathTracerGPU   # app (or VulkanPathTracer for just the lib)
```

Run from the repo root so asset paths resolve. Scenes live in `assets/scenes/` (the entry point
loads e.g. `ford.json`); saved screenshots go to `screenshots/`.

> Use `-j 4`, not a bare `-j` — a full-parallelism build OOM-kills this machine (exit 137).

## Conventions

- C++23. Prefer create-info structs for constructors (`XCreateInfo{ ... }`).
- **NEVER write code comments** unless explicitly asked — zero inline, block, header, or doc comments, even for new/complex/non-obvious code (shaders, math kernels, layout structs, Vulkan setup). Put explanations in the chat reply, never in the code.
- Diagnostics from clangd can be stale after renames/moves; confirm real errors with a build.
