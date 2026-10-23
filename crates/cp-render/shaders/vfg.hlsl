// Working frame <-> NVIDIA VFG packed RGB10A2 buffers (shared with CUDA).
//
// HDR content is encoded as PQ / BT.2020 (perceptually uniform, keeps wide
// gamut); SDR content as sRGB relative to SDR white, which is what the model
// was trained on. Bit layout matches DXGI R10G10B10A2 (R in the low bits).

#include "common.hlsli"

#define RS "RootConstants(num32BitConstants=8, b0)," \
           "DescriptorTable(SRV(t0))," \
           "DescriptorTable(UAV(u0))"

cbuffer Constants : register(b0)
{
    uint width;
    uint height;
    uint pitch;     // bytes per row
    uint hdr;       // 1 = PQ/BT.2020, 0 = sRGB
    float sdr_white; // SDR white in scRGB units
    float3 pad;
};

float3 linear_to_srgb3(float3 l)
{
    l = saturate(l);
    return lerp(1.055 * pow(l, 1.0 / 2.4) - 0.055, l * 12.92, step(l, 0.0031308));
}

float3 encode(float3 lin)
{
    if (hdr)
    {
        float3 nits = max(mul(BT709_TO_BT2020, lin * SCRGB_NITS), 0.0);
        return float3(nits_to_pq(nits.r), nits_to_pq(nits.g), nits_to_pq(nits.b));
    }
    return linear_to_srgb3(lin / sdr_white);
}

float3 decode(float3 e)
{
    if (hdr)
        return mul(BT2020_TO_BT709, pq_to_nits(e)) / SCRGB_NITS;
    return srgb_to_linear(e) * sdr_white;
}

#if ENCODE
Texture2D<float4> Source : register(t0);
RWByteAddressBuffer Packed : register(u0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;
    uint3 q = uint3(round(saturate(encode(Source.Load(int3(id, 0)).rgb)) * 1023.0));
    Packed.Store(id.y * pitch + id.x * 4, q.r | (q.g << 10) | (q.b << 20) | (3u << 30));
}
#else
ByteAddressBuffer Packed : register(t0);
RWTexture2D<float4> Output : register(u0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;
    uint v = Packed.Load(id.y * pitch + id.x * 4);
    float3 e = float3(v & 1023u, (v >> 10) & 1023u, (v >> 20) & 1023u) / 1023.0;
    Output[id] = float4(decode(e), 1.0);
}
#endif
