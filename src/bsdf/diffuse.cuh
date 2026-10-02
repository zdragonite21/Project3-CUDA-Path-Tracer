#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float pdf_diffuse(glm::vec3 wi) {
    return bx::cos_theta(wi) * INV_PI;
}

__device__ __forceinline__ glm::vec3 eval_diffuse(glm::vec3 wi, glm::vec3 color) {
    return color * INV_PI * max(0.f, bx::cos_theta(wi));
}

__device__ __forceinline__ BsdfSample sample_diffuse(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample;
    sample.wi = calculate_random_direction_in_cosine_hemisphere(rng);
    sample.pdf = pdf_diffuse(sample.wi);
    sample.f = eval_diffuse(sample.wi, m.color);
    sample.type = BxdfFlag::Diffuse;

    return sample;
}
