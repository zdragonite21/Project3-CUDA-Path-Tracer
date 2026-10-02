#pragma once

#include "fresnel.cuh"

#include "../scene_structs.h"
#include "../thrust_utils.h"
#include "bxdf_utils.cuh"
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
    float lambert = glm::abs(bx::cos_theta(sample.wi));
    sample.f = lambert > 0.f ? glm::vec3(fr) / lambert : glm::vec3(0);
    sample.type = BxdfFlag::Reflection | BxdfFlag::Specular;

    return sample;
}

__device__ __forceinline__ BsdfSample sample_conductor(glm::vec3 p, glm::vec3 wo, const Material& m,
                                                       RngEng& rng) {
    BsdfSample sample{};
    if (m.roughness == 0.f) {
        sample = sample_smooth_conductor(p, wo, m);
    }

    return sample;
}
