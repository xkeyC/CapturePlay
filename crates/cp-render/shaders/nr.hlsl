// Display-referred proxy and residual composite for DLSS Neural Rendering
// (NGX feature 18). The model works on an SDR, sRGB-coded image; HDR content
// is shown to it through a per-pixel invertible roll-off and only the model's
// bounded edit is applied back to the linear frame, so an unchanged proxy
// reproduces the original exactly.
//
// Math follows video2dlssnr (MIT, (c) 2026 Daniil Sokolyuk).

#include "common.hlsli"

float3 linear_to_srgb3(float3 l)
{
    l = saturate(l);
    return lerp(1.055 * pow(l, 1.0 / 2.4) - 0.055, l * 12.92, step(l, 0.0031308));
}

float luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

#if defined(FILL_DEPTH)
cbuffer Fill : register(b0)
{
    uint width;
    uint height;
    float value;
    uint pad;
};
RWTexture2D<float> Dst : register(u0);

[RootSignature("RootConstants(num32BitConstants=4, b0),DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x < width && id.y < height)
        Dst[id] = value;
}
#elif defined(FILL_MOTION)
cbuffer Fill : register(b0)
{
    uint width;
    uint height;
    float value;
    uint pad;
};
RWTexture2D<float2> Dst : register(u0);

[RootSignature("RootConstants(num32BitConstants=4, b0),DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x < width && id.y < height)
        Dst[id] = float2(0.0, 0.0);
}
#elif defined(ENCODE)
cbuffer C : register(b0)
{
    uint width;
    uint height;
    uint hdr;        // 1 = HDR content: roll the highlights off into the proxy
    float sdr_white; // SDR white in scRGB units (1.0 = 80 nits)
};
Texture2D<float4> Source : register(t0);
RWTexture2D<float4> Dst : register(u0);

[RootSignature("RootConstants(num32BitConstants=4, b0),DescriptorTable(SRV(t0)),DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;
    float3 c = max(Source.Load(int3(id, 0)).rgb / sdr_white, 0.0);
    if (hdr != 0)
    {
        float p = max(c.r, max(c.g, c.b));
        c = c / (1.0 + p);
    }
    Dst[id] = float4(linear_to_srgb3(saturate(c)), 1.0);
}
#else
cbuffer C : register(b0)
{
    uint width;
    uint height;
    uint hdr;
    float sdr_white;
    float detail;   // overall residual gain
    float colour;   // 0 = luminance-only edit, 1 = the model's colour too
    float2 pad;
};
Texture2D<float4> Source : register(t0);
Texture2D<float4> Encoded : register(t1);
RWTexture2D<float4> Dst : register(u0);

[RootSignature("RootConstants(num32BitConstants=8, b0),DescriptorTable(SRV(t0)),DescriptorTable(SRV(t1)),DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;
    float3 orig = Source.Load(int3(id, 0)).rgb;
    float3 o = srgb_to_linear(Encoded.Load(int3(id, 0)).rgb);
    float3 c = max(orig / sdr_white, 0.0);
    float3 result;
    if (hdr != 0)
    {
        float p = max(c.r, max(c.g, c.b));
        float3 i = c / (1.0 + p);
        float lI = luma(i);
        float lO = luma(o);
        float3 dHue = i * (lO / max(lI, 1e-4) - 1.0);
        float3 dFull = o - i;
        float3 delta = clamp(lerp(dHue, dFull, colour), -0.25, 0.25) * detail;
        result = clamp(c + (1.0 + p) * delta, 0.0, 64.0) * sdr_white;
    }
    else
    {
        float lo = luma(c);
        float ln = luma(o);
        float scale = ln / max(lo, 1e-4);
        float3 loC = c * scale;
        float3 cC = loC + (o - loC) * colour;
        result = (c + (cC - c) * detail) * sdr_white;
    }
    Dst[id] = float4(max(result, 0.0), 1.0);
}
#endif
