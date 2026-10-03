#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ float pdf_diffuse(glm::vec3 wi) {
    return bx::non_neg_cos(wi) * INV_PI;
}

__device__ __forceinline__ glm::vec3 eval_diffuse(glm::vec3 wi, const Lambertian& m) {
    return m.color * INV_PI * max(0.f, bx::cos_theta(wi));
}

__device__ __forceinline__ BsdfSample sample_diffuse(glm::vec3 p, const Lambertian& m,
                                                     RngEng& rng) {
    BsdfSample sample;
    sample.wi = calculate_random_direction_in_cosine_hemisphere(rng);
    sample.pdf = pdf_diffuse(sample.wi);
    sample.f = eval_diffuse(sample.wi, m);
    sample.type = BxdfFlag::Diffuse;

    return sample;
}
