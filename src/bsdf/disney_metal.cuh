#pragma once

#include "bsdf_structs.cuh"
#include "conductor.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

// Gulbrandsen 2014
__host__ __device__ __forceinline__ Conductor to_conductor(const DisneyMetal& m) {
    glm::vec3 r = glm::clamp(m.color, glm::vec3(0.f), glm::vec3(0.99f));
    glm::vec3 g = glm::clamp(m.edge_tint, glm::vec3(0.f), glm::vec3(1.f));
    glm::vec3 sr = glm::sqrt(r);

    glm::vec3 n_min = (1.f - r) / (1.f + r);
    glm::vec3 n_max = (1.f + sr) / (1.f - sr);
    glm::vec3 n = g * n_min + (1.f - g) * n_max;

    glm::vec3 np1 = n + 1.f;
    glm::vec3 nm1 = n - 1.f;
    glm::vec3 k2 = (r * np1 * np1 - nm1 * nm1) / (1.f - r);
    glm::vec3 k = glm::sqrt(glm::max(k2, glm::vec3(0.f)));

    return Conductor{n, k, m.roughness, m.anisotropic};
}

__device__ __forceinline__ BsdfEval eval_pdf_disney_metal(glm::vec3 wo, glm::vec3 wi,
                                                          const DisneyMetal& m) {
    return eval_pdf_conductor(wo, wi, to_conductor(m));
}

__device__ __forceinline__ BsdfSample sample_disney_metal(glm::vec3 wo, const DisneyMetal& m,
                                                          RngEng& rng) {
    return sample_conductor(wo, to_conductor(m), rng);
}
