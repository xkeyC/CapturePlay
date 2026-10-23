// Final pass: scale the working frame into the swapchain viewport and map it
// to the display (tone mapping to display peak, SDR white for SDR displays).
// The swapchain is always FP16 scRGB, so only these constants change when the
// window moves between SDR and HDR displays.

#include "common.hlsli"

#define RS "RootFlags(0)," \
           "RootConstants(num32BitConstants=20, b0)," \
           "DescriptorTable(SRV(t0))," \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"

cbuffer Constants : register(b0)
{
    float2 uv_scale;      // target pixel -> source uv
    float2 uv_offset;
    float2 src_size;
    float2 inv_src_size;
    uint sdr_display;     // 1 = display is SDR: normalize to SDR white
    float target_peak;    // nits: display peak (HDR) or SDR white (SDR)
    float src_peak;       // nits: mastering peak of the content
    float out_scale;      // nits -> output units (1/80 HDR, 1/sdr_white SDR)
    uint catmull_rom;
    float3 pad;
};

Texture2D<float4> Source : register(t0);
SamplerState Linear : register(s0);

// 9-tap Catmull-Rom using bilinear taps (Jimenez / TheRealMJP).
float3 sample_catmull_rom(float2 uv)
{
    float2 pos = uv * src_size;
    float2 tc1 = floor(pos - 0.5) + 0.5;
    float2 f = pos - tc1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 off12 = w2 / w12;
    float2 t0 = (tc1 - 1) * inv_src_size;
    float2 t3 = (tc1 + 2) * inv_src_size;
    float2 t12 = (tc1 + off12) * inv_src_size;
    float3 r = 0;
    r += Source.SampleLevel(Linear, float2(t0.x, t0.y), 0).rgb * w0.x * w0.y;
    r += Source.SampleLevel(Linear, float2(t12.x, t0.y), 0).rgb * w12.x * w0.y;
    r += Source.SampleLevel(Linear, float2(t3.x, t0.y), 0).rgb * w3.x * w0.y;
    r += Source.SampleLevel(Linear, float2(t0.x, t12.y), 0).rgb * w0.x * w12.y;
    r += Source.SampleLevel(Linear, float2(t12.x, t12.y), 0).rgb * w12.x * w12.y;
    r += Source.SampleLevel(Linear, float2(t3.x, t12.y), 0).rgb * w3.x * w12.y;
    r += Source.SampleLevel(Linear, float2(t0.x, t3.y), 0).rgb * w0.x * w3.y;
    r += Source.SampleLevel(Linear, float2(t12.x, t3.y), 0).rgb * w12.x * w3.y;
    r += Source.SampleLevel(Linear, float2(t3.x, t3.y), 0).rgb * w3.x * w3.y;
    return r;
}

float4 vs_main(uint id : SV_VertexID) : SV_Position
{
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
}

[RootSignature(RS)]
float4 ps_main(float4 pos : SV_Position) : SV_Target
{
    float2 uv = pos.xy * uv_scale + uv_offset;
    if (any(uv < 0.0) || any(uv > 1.0))
        return float4(0, 0, 0, 1);

    float3 c = catmull_rom ? sample_catmull_rom(uv) : Source.SampleLevel(Linear, uv, 0).rgb;
    float3 nits = max(c, -1e3) * SCRGB_NITS;

    // Hue-preserving tone map on the max channel.
    float m = max(max(nits.r, nits.g), nits.b);
    if (m > 0.0)
        nits *= bt2390_eetf(m, src_peak, target_peak) / m;

    float3 o = nits * out_scale;
    if (sdr_display)
        o = max(o, 0.0); // SDR outputs cannot show wide gamut anyway
    return float4(o, 1.0);
}
