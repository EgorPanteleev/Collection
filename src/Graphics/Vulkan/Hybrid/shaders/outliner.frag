#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D tracerImage;
layout(binding = 1) uniform usampler2D instanceIdImage;

void main() {
    const int thickness = 4;
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint instanceId  = texelFetch(instanceIdImage, p, 0).r;
    vec3 tracerColor = texelFetch(tracerImage, p, 0).rgb;
    ivec2 size = textureSize(instanceIdImage, 0);
    if (instanceId != 1) {
        outColor = vec4(tracerColor, 1.0);
        return;
    }
    bool edge = false;
    for (int y = -thickness; y <= thickness; ++y) {
        for (int x = -thickness; x <= thickness; ++x) {
            ivec2 q = p + ivec2(x, y);
            if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y)
            continue;
            if (x * x + y * y > thickness * thickness)
            continue;
            uint id = texelFetch(instanceIdImage, p + ivec2(x, y), 0).r;
            if (id != instanceId) {
                edge = true;
                break;
            }
        }
        if (edge) break;
    }

    if (edge) {
        outColor = vec4(vec3(1.0, 0.3, 0.0), 1.0);
    } else {
        outColor = vec4(tracerColor, 1.0);
    }
}