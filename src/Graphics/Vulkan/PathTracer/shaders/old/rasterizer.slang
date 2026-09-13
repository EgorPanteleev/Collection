struct MVP {                                          // per-frame transform UBO for the selection/ID rasterizer
    float4x4 model;                                   // model matrix (note: per-draw model comes from push constants below)
    float4x4 view;                                    // view matrix
    float4x4 proj;                                    // projection matrix
};                                                    // end MVP
[[vk::binding(0)]] ConstantBuffer<MVP> mvp;           // binding 0: the MVP constant buffer

struct PushConstants {                                // per-draw data pushed for each instance
    float4x4 model;                                   // this instance's model matrix
    uint     id;                                      // this instance's id, written to the ID target
};                                                    // end PushConstants
[[vk::push_constant]] PushConstants pc;               // the push-constant block

struct VSInput {                                      // vertex attributes (must match the vertex buffer layout)
    [[vk::location(0)]] float3 position;              // attribute 0: position
    [[vk::location(1)]] float2 uv;                    // attribute 1: texture coordinates (unused here)
    [[vk::location(2)]] float3 normal;                // attribute 2: normal (unused here)
    [[vk::location(3)]] float4 tangent;               // attribute 3: tangent (unused here)
};                                                    // end VSInput

[shader("vertex")]                                    // vertex stage
float4 vertMain(VSInput input) : SV_Position {        // transform the vertex to clip space
    return mul(mvp.proj, mul(mvp.view, mul(pc.model, float4(input.position, 1.0)))); // proj * view * model * position
}                                                     // end vertMain

[shader("fragment")]                                  // fragment stage
uint fragMain() : SV_Target0 {                        // output the instance id to the r32ui ID target
    return pc.id;                                     // write this draw's id
}                                                     // end fragMain
