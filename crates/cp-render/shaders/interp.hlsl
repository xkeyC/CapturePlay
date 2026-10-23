// Optical-flow frame interpolation (NVIDIA Optical Flow hardware + compute).
//
//   luma    : working scRGB frame -> 8-bit perceptual luma for NvOF input
//   densify : NvOF S10.5 grid vectors + cost -> float flow (working pixels)
//             with 3x3 vector-median outlier rejection and confidence
//   warp    : bidirectional backward warp at time t with forward/backward
//             consistency weighting (occlusion handling)
//
// Flow convention: F_ab(p) moves pixel p of A to p + F_ab(p) in B.

#include "common.hlsli"

#define RS "RootConstants(num32BitConstants=16, b0)," \
           "DescriptorTable(SRV(t0, numDescriptors=4))," \
           "DescriptorTable(UAV(u0))," \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"

cbuffer Constants : register(b0)
{
    uint2 dst_size;       // output extent of the current pass
    uint2 src_size;       // input extent
    float t;              // warp: interpolation position in (0,1)
    float flow_scale;     // densify: flow-res pixels -> working pixels
    float2 inv_grid;      // warp: 1 / (grid cell size in working px * grid extent)
    float2 inv_src;       // 1 / src_size
    uint grid_size;       // densify: NvOF grid (1,2,4)
    float max_cost;       // densify: cost treated as zero confidence
    float sigma;          // warp: consistency tolerance in pixels
    float pad0;
    float pad1;
    float pad2;
};

SamplerState Linear : register(s0);

// ---------------------------------------------------------------- luma
Texture2D<float4> LumaSrc : register(t0);
RWTexture2D<unorm float> LumaOut : register(u0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void luma(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= dst_size))
        return;
    // Box-filtered downscale (src_size / dst_size is 1 or 2).
    float2 uv = (float2(id) + 0.5) / float2(dst_size);
    float3 c = LumaSrc.SampleLevel(Linear, uv, 0).rgb;
    float nits = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 0.0) * SCRGB_NITS;
    // PQ is perceptually uniform from 0.1 to 10k nits; stretch the useful
    // 0..1000 nit range over the 8-bit code space.
    LumaOut[id] = saturate(nits_to_pq(nits) / 0.7518);
}

// ---------------------------------------------------------------- densify
Texture2D<int2> FlowIn : register(t0);
Texture2D<uint> CostIn : register(t1);
RWTexture2D<float4> FlowOut : register(u0);

float2 load_flow(int2 p)
{
    p = clamp(p, int2(0, 0), int2(src_size) - 1);
    return float2(FlowIn.Load(int3(p, 0))) * (1.0 / 32.0);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void densify(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= dst_size))
        return;
    // 3x3 vector median: the candidate minimizing summed distance to others.
    float2 v[9];
    [unroll] for (int i = 0; i < 9; ++i)
        v[i] = load_flow(int2(id) + int2(i % 3 - 1, i / 3 - 1));
    float best = 1e30;
    float2 med = v[4];
    [unroll] for (int a = 0; a < 9; ++a)
    {
        float d = 0;
        [unroll] for (int b = 0; b < 9; ++b)
            d += length(v[a] - v[b]);
        if (d < best)
        {
            best = d;
            med = v[a];
        }
    }
    float cost = float(CostIn.Load(int3(id, 0)));
    float conf = saturate(1.0 - cost / max_cost);
    FlowOut[id] = float4(med * flow_scale, conf, 0);
}

// ---------------------------------------------------------------- warp
Texture2D<float4> FrameA : register(t0);
Texture2D<float4> FrameB : register(t1);
Texture2D<float4> FlowAB : register(t2);
Texture2D<float4> FlowBA : register(t3);
RWTexture2D<float4> WarpOut : register(u0);

float3 sample_flow(Texture2D<float4> f, float2 pos_px)
{
    return f.SampleLevel(Linear, pos_px * inv_grid, 0).xyz;
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void warp(uint2 id : SV_DispatchThreadID)
{
    if (any(id >= dst_size))
        return;
    float2 p = float2(id) + 0.5;

    // Solve q = p - t * F_ab(q) and r = p - (1-t) * F_ba(r) (relaxed Picard).
    float2 q = p, r = p;
    [unroll] for (int i = 0; i < 6; ++i)
    {
        q = lerp(q, p - t * sample_flow(FlowAB, q).xy, 0.5 + 0.5 * (i > 1));
        r = lerp(r, p - (1.0 - t) * sample_flow(FlowBA, r).xy, 0.5 + 0.5 * (i > 1));
    }
    float3 fq = sample_flow(FlowAB, q);
    float3 fr = sample_flow(FlowBA, r);

    // Forward/backward consistency: occluded pixels disagree.
    float ea = length(fq.xy + sample_flow(FlowBA, q + fq.xy).xy);
    float eb = length(fr.xy + sample_flow(FlowAB, r + fr.xy).xy);
    float s2 = sigma * sigma;
    float wa = (1.0 - t) * exp(-ea * ea / s2) * (0.25 + 0.75 * fq.z);
    float wb = t * exp(-eb * eb / s2) * (0.25 + 0.75 * fr.z);

    float3 ca = FrameA.SampleLevel(Linear, q * inv_src, 0).rgb;
    float3 cb = FrameB.SampleLevel(Linear, r * inv_src, 0).rgb;
    float3 c;
    if (wa + wb < 1e-4)
        c = lerp(ca, cb, t); // both unreliable: plain cross-fade
    else
        c = (wa * ca + wb * cb) / (wa + wb);
    WarpOut[id] = float4(c, 1.0);
}
