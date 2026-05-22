#ifndef LAYOUT_GLSL
#define LAYOUT_GLSL

#include "bvh.glsl"

layout(push_constant) uniform PushConstants {
    uint frame;
    uint spp;
    uint minDepth;
    uint maxDepth;
    uint displayMode;
} pc;

layout(binding = 0) readonly uniform Camera {
    vec4 pos;
    mat4 invViewProj;
} camera;

layout(binding = 1) readonly buffer TriangleBuffer {
    PrecomputedTriangle data[];
} triangleBuffer;

layout(binding = 2) readonly buffer TriangleExtraBuffer {
    TriangleExtra data[];
} triangleExtraBuffer;

layout(binding = 3) readonly buffer NodeBuffer {
    Node data[];
} nodeBuffer;

layout(binding = 4) readonly buffer TLASNodeBuffer_ {
    Node data[];
} TLASNodeBuffer;

layout(binding = 5) readonly buffer InstanceBuffer {
    Instance data[];
} instanceBuffer;

layout(binding = 6) readonly uniform DirectLight {
    vec4 dir;
    float intensity;
    float pad[3];
} directLight;

layout(binding = 7, rgba8) uniform image2D outputImage;

layout(binding = 8) uniform sampler2D colorImage;
layout(binding = 9) uniform sampler2D depthImage;
layout(binding = 10) uniform sampler2D normalImage;

layout(binding = 11) uniform sampler2D textures[];

BLASHit intersect(Ray ray, uint rootNode, uint baseTri, float eps) {
    BLASHit closestHit = BLASHit(UINT_MAX, FLT_MAX, 0, 0);
    uint stack[8];
    uint sp = 0;
    stack[sp++] = rootNode;
    while (sp != 0) {
        uint nodeIdx = stack[--sp];
        Node node = nodeBuffer.data[nodeIdx];
        float tNode = intersect(node.bbox, ray);
        if (tNode == FLT_MAX || tNode >= closestHit.t) continue;
        uint primCount = primCount(node.index);
        if (primCount != 0) {
            uint id = baseTri + id(node.index);
            for (uint i = 0; i < primCount; ++i) {
                PrecomputedTriangle tri = triangleBuffer.data[id + i];
                TriHit triHit = intersect(tri, ray, eps);
                if (triHit.t > eps && triHit.t < closestHit.t) {
                    closestHit = BLASHit(id + i, triHit.t, triHit.u, triHit.v);
                }
            }
        } else {
            uint leftIdx  = rootNode + id(node.index);
            uint rightIdx = leftIdx + 1;
            Node left  = nodeBuffer.data[leftIdx];
            Node right = nodeBuffer.data[rightIdx];
            float tLeft  = intersect(left.bbox, ray);
            float tRight = intersect(right.bbox, ray);
            bool hitLeft  = (tLeft  != FLT_MAX) && (tLeft  < closestHit.t);
            bool hitRight = (tRight != FLT_MAX) && (tRight < closestHit.t);
            if (hitLeft && hitRight) {
                if (tLeft < tRight) {
                    stack[sp++] = rightIdx;
                    stack[sp++] = leftIdx;
                } else {
                    stack[sp++] = leftIdx;
                    stack[sp++] = rightIdx;
                }
            } else if (hitLeft) {
                stack[sp++] = leftIdx;
            } else if (hitRight) {
                stack[sp++] = rightIdx;
            }
        }
    }
    return closestHit;
}

Hit intersect(Ray ray, float eps) {
    Hit closestHit = Hit(UINT_MAX, UINT_MAX, FLT_MAX, 0, 0);
    uint stack[4];
    uint sp = 0;
    stack[sp++] = 0;
    while (sp != 0) {
        uint nodeIdx = stack[--sp];
        Node node = TLASNodeBuffer.data[nodeIdx];
        float tNode = intersect(node.bbox, ray);
        if (tNode == FLT_MAX || tNode >= closestHit.t) continue;
        uint primCount = primCount(node.index);
        if (primCount != 0) {
            uint id = id(node.index);
            for (uint i = 0; i < primCount; ++i) {
                Instance instance = instanceBuffer.data[id + i];
                vec3 localPos = vec3(instance.invModel * vec4(ray.pos, 1.0));
                vec3 localDir = vec3(instance.invModel * vec4(ray.dir, 0.0));
                Ray localRay = makeRay(localPos, localDir, 0.0, FLT_MAX);
                BLASHit instanceHit = intersect(localRay, instance.baseNode, instance.baseTri, eps);
                if (instanceHit.t > eps && instanceHit.t < closestHit.t) {
                    closestHit = Hit(instanceHit.id, id + i, instanceHit.t, instanceHit.u, instanceHit.v);
                }
            }
        } else {
            uint leftIdx  = id(node.index);
            uint rightIdx = leftIdx + 1;
            Node left  = TLASNodeBuffer.data[leftIdx];
            Node right = TLASNodeBuffer.data[rightIdx];
            float tLeft  = intersect(left.bbox, ray);
            float tRight = intersect(right.bbox, ray);
            bool hitLeft  = (tLeft  != FLT_MAX) && (tLeft  < closestHit.t);
            bool hitRight = (tRight != FLT_MAX) && (tRight < closestHit.t);
            if (hitLeft && hitRight) {
                if (tLeft < tRight) {
                    stack[sp++] = rightIdx;
                    stack[sp++] = leftIdx;
                } else {
                    stack[sp++] = leftIdx;
                    stack[sp++] = rightIdx;
                }
            } else if (hitLeft) {
                stack[sp++] = leftIdx;
            } else if (hitRight) {
                stack[sp++] = rightIdx;
            }
        }
    }
    return closestHit;
}

#endif
