// Shared color math. Reference implementation and tests: cp-core/src/color.rs.
// Working space: linear scRGB (BT.709 primaries, 1.0 = 80 nits).

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

static const float SCRGB_NITS = 80.0;

#define TF_SRGB    0
#define TF_GAMMA22 1
#define TF_BT1886  2
#define TF_PQ      3
#define TF_HLG     4

static const float3x3 BT2020_TO_BT709 = {
     1.660491, -0.587641, -0.072850,
    -0.124550,  1.132900, -0.008349,
    -0.018151, -0.100579,  1.118730,
};

static const float3x3 BT709_TO_BT2020 = {
    0.627404, 0.329283, 0.043313,
    0.069097, 0.919540, 0.011362,
    0.016391, 0.088013, 0.895595,
};

static const float PQ_M1 = 2610.0 / 16384.0;
static const float PQ_M2 = 2523.0 / 4096.0 * 128.0;
static const float PQ_C1 = 3424.0 / 4096.0;
static const float PQ_C2 = 2413.0 / 4096.0 * 32.0;
static const float PQ_C3 = 2392.0 / 4096.0 * 32.0;

float3 pq_to_nits(float3 e)
{
    float3 p = pow(saturate(e), 1.0 / PQ_M2);
    float3 num = max(p - PQ_C1, 0.0);
    float3 den = PQ_C2 - PQ_C3 * p;
    return 10000.0 * pow(num / den, 1.0 / PQ_M1);
}

float nits_to_pq(float nits)
{
    float y = pow(saturate(nits / 10000.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), PQ_M2);
}

float3 hlg_to_scene(float3 e)
{
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    e = saturate(e);
    float3 lo = e * e / 3.0;
    float3 hi = (exp((e - c) / a) + b) / 12.0;
    return lerp(hi, lo, step(e, 0.5));
}

// BT.2100 HLG OOTF on linear BT.2020 scene light, output in nits.
float3 hlg_ootf(float3 scene, float peak_nits, float gamma)
{
    float ys = dot(scene, float3(0.2627, 0.6780, 0.0593));
    return scene * (peak_nits * pow(max(ys, 1e-6), gamma - 1.0));
}

float3 srgb_to_linear(float3 c)
{
    c = saturate(c);
    return lerp(pow((c + 0.055) / 1.055, 2.4), c / 12.92, step(c, 0.04045));
}

// BT.2390 EETF in the PQ domain; identity when the target covers the source.
float bt2390_eetf(float nits, float src_peak, float dst_peak)
{
    if (dst_peak >= src_peak)
        return min(nits, src_peak);
    float src_max = nits_to_pq(src_peak);
    float e1 = nits_to_pq(nits) / src_max;
    float max_lum = nits_to_pq(dst_peak) / src_max;
    float ks = 1.5 * max_lum - 0.5;
    float e2 = e1;
    if (e1 >= ks)
    {
        float t = min((e1 - ks) / (1.0 - ks), 1.0);
        float t2 = t * t, t3 = t2 * t;
        e2 = (2 * t3 - 3 * t2 + 1) * ks + (t3 - 2 * t2 + t) * (1 - ks) + (-2 * t3 + 3 * t2) * max_lum;
    }
    return pq_to_nits(e2 * src_max).x;
}

#endif
