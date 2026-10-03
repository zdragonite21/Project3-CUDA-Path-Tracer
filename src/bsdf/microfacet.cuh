#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bxdf_utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float ggx_lambda(glm::vec3 w, float ax, float ay) {
    float x = w.x * ax;
    float y = w.y * ay;
    float q = sqrtf(1.f + (x * x + y * y) / (w.z * w.z));

    return (q - 1.f) / 2.f;
}

__device__ __forceinline__ glm::vec2 ggx_alpha(float anis, float rough) {
    float aspect = sqrtf(1.f - 0.9f * anis);
    const float amin = 0.0001f;
    float ax = glm::clamp(rough * rough / aspect, amin, 1.f);
    float ay = glm::clamp(rough * rough * aspect, amin, 1.f);
    return glm::vec2(ax, ay);
}

__device__ __forceinline__ float ggx_d(glm::vec3 wh, float ax, float ay) {
    float x = wh.x * wh.x / (ax * ax) + wh.y * wh.y / (ay * ay) + wh.z * wh.z;
    float denom = PI * ax * ay * x * x;
    return 1.f / denom;
}

__device__ __forceinline__ float ggx_g1(glm::vec3 w, float ax, float ay) {
    return 1.f / (1.f + ggx_lambda(w, ax, ay));
}

__device__ __forceinline__ float ggx_g2(glm::vec3 wi, glm::vec3 wo, float ax, float ay) {
    return 1.f / (1.f + ggx_lambda(wi, ax, ay) + ggx_lambda(wo, ax, ay));
}

__device__ __forceinline__ float pdf_ggx(glm::vec3 wi, glm::vec3 wo, float anis, float rough) {
    if (bx::cos_theta(wi) * bx::cos_theta(wo) <= 0.f) {
        // no transmission
        return 0.f;
    }

    float len = glm::length(wi + wo);
    if (len <= 0) {
        return 0.f;
    }
    glm::vec3 wh = (wi + wo) / len;
    wh = wh.z < 0 ? -wh : wh;

    float wo_cos = bx::abs_cos(wo);
    if (wo_cos <= 0.f) {
        return 0.f;
    }

    glm::vec2 a = ggx_alpha(anis, rough);
    float d = ggx_d(wh, a.x, a.y);
    float g1 = ggx_g1(wo, a.x, a.y);

    return d * g1 / (4.f * wo_cos);
}

__device__ __forceinline__ float eval_ggx_dg(glm::vec3 wi, glm::vec3 wo, float anis, float rough) {
    if (bx::cos_theta(wi) * bx::cos_theta(wo) <= 0.f) {
        // no transmission
        return 0.f;
    }

    float len = glm::length(wi + wo);
    if (len <= 0) {
        return 0.f;
    }
    glm::vec3 wh = (wi + wo) / len;
    wh = wh.z < 0 ? -wh : wh;

    glm::vec2 a = ggx_alpha(anis, rough);

    float d = ggx_d(wh, a.x, a.y);
    float g = ggx_g2(wi, wo, a.x, a.y);

    return d * g / (4.f * bx::abs_cos(wo));
}

// Dupuy & Benyoub 2023
__device__ __forceinline__ glm::vec3 sample_ggx_vndf(glm::vec3 wo, float ax, float ay,
                                                     RngEng& rng) {
    glm::vec3 wo_std = glm::normalize(glm::vec3(wo.x * ax, wo.y * ay, wo.z));

    glm::vec3 c = sample_uniform_sphere_cap(rng, wo_std.z);

    glm::vec3 wh_std = c + wo_std;
    return glm::normalize(glm::vec3(wh_std.x * ax, wh_std.y * ay, wh_std.z));
}

__device__ __forceinline__ BsdfSample sample_ggx(glm::vec3 p, glm::vec3 wo, float anis, float rough,
                                                 RngEng& rng) {
    BsdfSample sample{};
    if (bx::cos_theta(wo) <= 0.f) {
        return sample;
    }

    glm::vec2 a = ggx_alpha(anis, rough);
    glm::vec3 wh = sample_ggx_vndf(wo, a.x, a.y, rng);
    glm::vec3 wi = glm::reflect(-wo, wh);
    if (bx::cos_theta(wi) <= 0.f) {
        return sample;
    }

    sample.wi = wi;
    sample.pdf = pdf_ggx(wi, wo, anis, rough);
    sample.f = glm::vec3(eval_ggx_dg(wi, wo, anis, rough));
    sample.type = BxdfFlag::Glossy | BxdfFlag::Reflection;

    return sample;
}
