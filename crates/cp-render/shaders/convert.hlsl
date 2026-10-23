// Captured Y'CbCr / RGB planes -> linear scRGB working frame.
// Variants (compile-time): PLANAR_420 (NV12/P010), PACKED_YUYV, PACKED_UYVY, PACKED_BGRA.

#include "common.hlsli"

#define RS "RootConstants(num32BitConstants=24, b0)," \
           "DescriptorTable(SRV(t0, numDescriptors=2))," \
           "DescriptorTable(UAV(u0))," \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"

cbuffer Constants : register(b0)
{
    float4 row0;          // R' = dot(row0.xyz, ycc) + row0.w
    float4 row1;
    float4 row2;
    uint transfer;        // TF_*
    uint primaries_2020;  // 1 = convert BT.2020 -> BT.709
    float sdr_white;      // SDR white in scRGB units (nits / 80)
    float hlg_peak;       // nits
    float hlg_gamma;
    uint width;
    uint height;
    uint pad0;
    float2 chroma_texel;  // 1 / chroma plane size
    float2 pad1;
};

#if PLANAR_420
Texture2D<float> LumaTex : register(t0);
Texture2D<float2> ChromaTex : register(t1);
#else
Texture2D<float4> PackedTex : register(t0);
#endif
RWTexture2D<float4> Output : register(u0);
SamplerState Linear : register(s0);

float3 to_scrgb(float3 rgbp)
{
    float3 lin;
    if (transfer == TF_PQ)
        lin = pq_to_nits(rgbp) / SCRGB_NITS;
    else if (transfer == TF_HLG)
        lin = hlg_ootf(hlg_to_scene(rgbp), hlg_peak, hlg_gamma) / SCRGB_NITS;
    else if (transfer == TF_GAMMA22)
        lin = pow(saturate(rgbp), 2.2) * sdr_white;
    else if (transfer == TF_BT1886)
        lin = pow(saturate(rgbp), 2.4) * sdr_white;
    else
        lin = srgb_to_linear(rgbp) * sdr_white;

    if (primaries_2020)
        lin = mul(BT2020_TO_BT709, lin);
    return lin;
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height)
        return;

#if PACKED_BGRA
    float3 rgbp = PackedTex.Load(int3(id, 0)).rgb;
#else
    float y;
    float2 c;
#if PLANAR_420
    y = LumaTex.Load(int3(id, 0));
    // Chroma sited left (co-sited horizontally), centered vertically.
    float2 cpos = float2(id.x * 0.5 + 0.5, (id.y + 0.5) * 0.5);
    c = ChromaTex.SampleLevel(Linear, cpos * chroma_texel, 0);
#else
    // One texel = two pixels: YUYV -> (Y0, U, Y1, V); UYVY -> (U, Y0, V, Y1).
    uint2 tp = uint2(id.x >> 1, id.y);
    float4 t = PackedTex.Load(int3(tp, 0));
    float4 tn = PackedTex.Load(int3(min(tp.x + 1, (width >> 1) - 1), tp.y, 0));
    bool odd = (id.x & 1) != 0;
#if PACKED_YUYV
    y = odd ? t.b : t.r;
    c = odd ? 0.5 * (t.ga + tn.ga) : t.ga;
#else
    y = odd ? t.a : t.g;
    c = odd ? 0.5 * (t.rb + tn.rb) : t.rb;
#endif
#endif
    float3 ycc = float3(y, c);
    float3 rgbp = float3(dot(row0.xyz, ycc) + row0.w, dot(row1.xyz, ycc) + row1.w, dot(row2.xyz, ycc) + row2.w);
#endif

    Output[id] = float4(to_scrgb(rgbp), 1.0);
}
