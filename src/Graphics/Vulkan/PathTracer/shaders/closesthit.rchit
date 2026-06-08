#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct Vertex {
    vec3 pos;
    vec2 texCoord;
    vec3 normal;
    vec4 tangent;
};
struct MeshInfo {
    uint64_t vertexAddress;
    uint64_t indexAddress;
};
struct InstanceData {
    uint meshID;
    uint textureID;
    uint pad[2];
};
layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};
layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint indices[];
};
layout(set = 0, binding = 3, scalar) readonly buffer MeshInfoBuffer {
    MeshInfo meshes[];
};
layout(set = 0, binding = 4, scalar) readonly buffer InstanceBuffer {
    InstanceData instances[];
};
layout(binding = 5) uniform sampler2D textures[];
layout(location = 0) rayPayloadInEXT vec3 payloadColor;

hitAttributeEXT vec2 hitAttrib;

void main() {
    InstanceData instance = instances[gl_InstanceCustomIndexEXT];
    MeshInfo mesh = meshes[instance.meshID];
    VertexBuffer vertexBuffer = VertexBuffer(mesh.vertexAddress);
    IndexBuffer  indexBuffer  = IndexBuffer(mesh.indexAddress);

    uint i0 = indexBuffer.indices[gl_PrimitiveID * 3 + 0];
    uint i1 = indexBuffer.indices[gl_PrimitiveID * 3 + 1];
    uint i2 = indexBuffer.indices[gl_PrimitiveID * 3 + 2];

    vec2 uv0 = vertexBuffer.vertices[i0].texCoord;
    vec2 uv1 = vertexBuffer.vertices[i1].texCoord;
    vec2 uv2 = vertexBuffer.vertices[i2].texCoord;

    float u = hitAttrib.x; float v = hitAttrib.y; float w = 1 - u - v;
    vec2 uv = w * uv0 + u * uv1 + v * uv2;

    payloadColor = texture(textures[nonuniformEXT(instance.textureID)], uv).rgb;;//vec3(1.0, 1.0, 1.0);
}