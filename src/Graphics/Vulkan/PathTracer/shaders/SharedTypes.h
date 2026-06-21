#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#ifdef __cplusplus
  #include <glm/glm.hpp>
  #include <cstdint>
  #define float2   glm::vec2
  #define float3   glm::vec3
  #define float4   glm::vec4
  #define float4x4 glm::mat4
  #define uint     uint32_t
  #define SLANG_PUBLIC
  #define GPU_PTR(T) uint64_t
namespace crv::graphics::vulkan {
#else
  #define SLANG_PUBLIC public
  #define GPU_PTR(T) T*
#endif

SLANG_PUBLIC struct Vertex {
    SLANG_PUBLIC float3 pos;
    SLANG_PUBLIC float2 texCoord;
    SLANG_PUBLIC float3 normal;
    SLANG_PUBLIC float4 tangent;
};

SLANG_PUBLIC struct AliasEntry {
    SLANG_PUBLIC float prob;
    SLANG_PUBLIC uint  alias;
};

SLANG_PUBLIC struct MeshGPU {
    SLANG_PUBLIC GPU_PTR(Vertex) vertices;
    SLANG_PUBLIC GPU_PTR(uint)   indices;
    SLANG_PUBLIC uint            indexCount;
    SLANG_PUBLIC float           area;
    SLANG_PUBLIC uint64_t        aliasAddress;
};

SLANG_PUBLIC struct InstanceGPU {
    SLANG_PUBLIC float4x4 model;
    SLANG_PUBLIC uint     meshIndex;
    SLANG_PUBLIC uint     materialIndex;
};

SLANG_PUBLIC struct MaterialGPU {
    SLANG_PUBLIC float3 baseColor;
    SLANG_PUBLIC float3 emissionColor;
    SLANG_PUBLIC float  luminance;
    SLANG_PUBLIC float  metalness;
    SLANG_PUBLIC float  roughness;
    SLANG_PUBLIC float  ior;
    SLANG_PUBLIC float  specular;
    SLANG_PUBLIC float  transmission;
    SLANG_PUBLIC float  clearcoat;
    SLANG_PUBLIC float  clearcoatRoughness;
    SLANG_PUBLIC uint   baseColorTexIndex;
    SLANG_PUBLIC uint   normalTexIndex;
};

#ifdef __cplusplus
static_assert(sizeof(Vertex)       == 48, "Vertex layout must match the shader scalar buffer layout");
static_assert(sizeof(AliasEntry)   ==  8, "AliasEntry layout must match the shader scalar buffer layout");
static_assert(sizeof(MeshGPU)     == 32, "MeshGPU layout must match the shader scalar buffer layout");
static_assert(sizeof(InstanceGPU)  == 72, "InstanceGPU layout must match the shader scalar buffer layout");
static_assert(sizeof(MaterialGPU) == 64, "MaterialGPU layout must match the shader scalar buffer layout");
} // namespace crv::graphics::vulkan
  #undef float2
  #undef float3
  #undef float4
  #undef float4x4
  #undef uint
#endif
#undef SLANG_PUBLIC
#undef GPU_PTR

#endif // SHARED_TYPES_H
