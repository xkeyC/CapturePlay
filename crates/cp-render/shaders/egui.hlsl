// egui overlay drawn straight into the scRGB back buffer. egui produces
// sRGB-encoded premultiplied colors; they are linearized and scaled to the
// UI white level so the overlay looks right on both SDR and HDR displays.

#include "common.hlsli"

#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
           "RootConstants(num32BitConstants=4, b0)," \
           "DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_PIXEL)," \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP, visibility=SHADER_VISIBILITY_PIXEL)"

cbuffer Constants : register(b0)
{
    float2 screen_size_points;
    float white_scale; // UI white in output units
    float pad;
};

Texture2D<float4> Tex : register(t0); // *_UNORM_SRGB view: sampled as linear
SamplerState Linear : register(s0);

struct VsIn
{
    float2 pos : POSITION;
    float2 uv : TEXCOORD;
    float4 color : COLOR; // R8G8B8A8_UNORM, sRGB-encoded, premultiplied
};

struct VsOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};

[RootSignature(RS)]
VsOut vs_main(VsIn i)
{
    VsOut o;
    o.pos = float4(2.0 * i.pos.x / screen_size_points.x - 1.0, 1.0 - 2.0 * i.pos.y / screen_size_points.y, 0.0, 1.0);
    o.uv = i.uv;
    o.color = float4(srgb_to_linear(i.color.rgb), i.color.a);
    return o;
}

[RootSignature(RS)]
float4 ps_main(VsOut i) : SV_Target
{
    float4 c = i.color * Tex.Sample(Linear, i.uv);
    return float4(c.rgb * white_scale, c.a);
}
