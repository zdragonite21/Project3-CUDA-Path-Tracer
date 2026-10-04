#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ BsdfEval eval_pdf_disney_diffuse(glm::vec3 wo, glm::vec3 wi,
                                                            glm::vec3 base_color, float roughness,
                                                            float subsurface) {
    BsdfEval eval{};
    if (bx::cos_theta(wo) * bx::cos_theta(wi) <= 0.f) {
        return eval;
    }

    glm::vec3 h = wo + wi;
    float len2 = glm::dot(h, h);
    if (len2 <= 0.f) {
        return eval;
    }
    h *= rsqrtf(len2);
    h = h.z < 0.f ? -h : h;

    return eval;
}
__device__ __forceinline__ BsdfSample sample_disney_diffuse(glm::vec3 wo, glm::vec3 base_color,
                                                            float roughness, float subsurface,
                                                            RngEng& rng);
