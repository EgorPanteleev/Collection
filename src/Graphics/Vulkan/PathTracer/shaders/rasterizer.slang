struct MVP {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
[[vk::binding(0)]] ConstantBuffer<MVP> mvp;

struct InstanceData {
    float4x4 model;
};
[[vk::binding(1)]] ConstantBuffer<InstanceData> instance;

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float3 normal;
    [[vk::location(3)]] float4 tangent;
};

[shader("vertex")]
float4 vertMain(VSInput input) : SV_Position {
    return mul(mvp.proj, mul(mvp.view, mul(instance.model, float4(input.position, 1.0))));
}

[shader("fragment")]
uint fragMain() : SV_Target0 {
    return 1u;
}
