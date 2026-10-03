#pragma once

#include "fresnel.cuh"

#include "../scene_structs.h"
#include "../thrust_utils.h"
#include "bxdf_utils.cuh"
#include "microfacet.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// assumes the medium is air and rgb approx, not spectral
__device__ __forceinline__ BsdfSample sample_smooth_conductor(glm::vec3 p, glm::vec3 wo,
                                                              const Material& m) {
    BsdfSample sample;

    sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
    sample.pdf = 1.f;
    // eta_i for air is 1
    glm::vec3 fr = fresnel_conductor_eval(bx::cos_theta(sample.wi), glm::vec3(1), m.eta, m.k);
    sample.f = glm::vec3(fr);
    sample.type = BxdfFlag::Reflection | BxdfFlag::Specular;

    return sample;
}

__device__ __forceinline__ float pdf_conductor(glm::vec3 wi, glm::vec3 wo, const Material& m) {
    if (m.roughness == 0.f) {
        return 0.f;
    }

    return pdf_ggx(wi, wo, m.anisotropy, m.roughness);
}

__device__ __forceinline__ glm::vec3 eval_conductor(glm::vec3 wi, glm::vec3 wo, const Material& m) {
    if (m.roughness == 0.f) {
        return glm::vec3(0);
    }

    float ggx_dg = eval_ggx_dg(wi, wo, m.anisotropy, m.roughness);
    if (ggx_dg == 0.f) {
        return glm::vec3(0.f);
    }

    glm::vec3 wh = glm::normalize(wi + wo);
    return fresnel_conductor_eval(glm::dot(wo, wh), glm::vec3(1.f), m.eta, m.k) * ggx_dg;
}

__device__ __forceinline__ BsdfSample sample_conductor(glm::vec3 p, glm::vec3 wo, const Material& m,
                                                       RngEng& rng) {
    if (m.roughness == 0.f) {
        return sample_smooth_conductor(p, wo, m);
    }

    BsdfSample sample = sample_ggx(p, wo, m.anisotropy, m.roughness, rng);
    if (sample.type == BxdfFlag::Unset) {
        return sample;
    }
    glm::vec3 wh = glm::normalize(sample.wi + wo);
    sample.f *= fresnel_conductor_eval(glm::dot(wo, wh), glm::vec3(1), m.eta, m.k);

    return sample;
}
