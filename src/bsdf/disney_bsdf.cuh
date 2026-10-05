#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ BsdfEval eval_pdf_disney_bsdf(glm::vec3 wo, glm::vec3 wi,
                                                         const DisneyBsdf& m) {
    BsdfEval eval{};
    return eval;
}

__device__ __forceinline__ BsdfSample sample_disney_bsdf(glm::vec3 wo, const DisneyBsdf& m,
                                                         RngEng& rng) {
    return BsdfSample{};
}
