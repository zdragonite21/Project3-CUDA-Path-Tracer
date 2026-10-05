#pragma once

#include "../math_utils.h"
#include "../sampling.cuh"

#include "disney_clearcoat.cuh"
#include "disney_diffuse.cuh"
#include "disney_glass.cuh"
#include "disney_metal.cuh"
#include "disney_sheen.cuh"

#include "bsdf_structs.cuh"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __forceinline__ BsdfEval eval_pdf_disney_bsdf(glm::vec3 wo, glm::vec3 wi,
                                                         const DisneyBsdf& m) {
    BsdfEval eval{};

    bool inside = bx::cos_theta(wi) < 0.f;

    if (!inside && m.specular_transmission < 1.f && m.metallic < 1.f) {
        BsdfEval diffuse_eval =
            eval_pdf_disney_diffuse(wo, wi, DisneyDiffuse{m.color, m.roughness, m.subsurface});
        eval.f += (1.f - m.specular_transmission) * (1.f - m.metallic) * diffuse_eval.f;
    }
    if (!inside && m.metallic < 1.f && m.sheen > 0.f) {
        BsdfEval sheen_eval = eval_pdf_disney_sheen(wo, wi, DisneySheen{m.color, m.sheen_tint});
        eval.f += (1.f - m.metallic) * m.sheen * sheen_eval.f;
    }
    if (!inside && !(m.metallic == 0.f && m.specular_transmission == 0.f)) {
        BsdfEval metal_eval = eval_pdf_disney_metal(wo, wi, DisneyMetal{m.color, m.edge_tint, m.roughness, m.anisotropic});
        eval.f += (1.f - m.specular_transmission * (1.f - m.metallic)) * metal_eval.f; 
    }
    if (!inside && m.clearcoat > 0.f) {
        BsdfEval clearcoat_eval = eval_pdf_disney_clearcoat(wo, wi, DisneyClearcoat{m.clearcoat_gloss});
        eval.f += 0.25f * m.clearcoat * clearcoat_eval.f;
    }
    if (m.metallic < 1.f && m.specular_transmission > 0.f) {
        BsdfEval glass_eval = eval_pdf_disney_glass(wo, wi, DisneyGlass{m.color, m.roughness, m.ior});
        eval.f += (1.f - m.metallic) * m.specular_transmission * glass_eval.f;
    }

    return eval;
}

__device__ __forceinline__ BsdfSample sample_disney_bsdf(glm::vec3 wo, const DisneyBsdf& m,
                                                         RngEng& rng) {
    return BsdfSample{};
}
