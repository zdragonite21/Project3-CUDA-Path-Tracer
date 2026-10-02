#pragma once

#include "fresnel.cuh"

#include "../scene_structs.h"
#include "../thrust_utils.h"
#include "bxdf_utils.cuh"
#include <cuda_runtime.h>
#include <glm/glm.hpp>


// assumes the medium is air and not spectral
__device__ __forceinline__ BsdfSample sample_smooth_dielectric(glm::vec3 p, glm::vec3 wo,
                                                               const Material& m, RngEng& rng) {
    BsdfSample sample;
    UnifDist<float> u01(0, 1);

    // eta_i = 1.f because we assume air here
    float r = fresnel_dielectric_eval(bx::cos_theta(wo), 1.f, m.ior);
    float t = 1.f - r;

    if (u01(rng) < r / (r + t)) {
        // sample perfect specular reflection
        sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
        sample.pdf = r / (r + t);
        sample.f = glm::vec3(r);
        sample.type = BxdfFlag::Reflection;
    } else {
        // sample perfect specular transmission
        bool entering = bx::cos_theta(wo) > 0;
        float eta_i = entering ? 1.f : m.ior;
        float eta_t = entering ? m.ior : 1.f;
        float eta = eta_i / eta_t;
        glm::vec3 wi;
        if (!bx::refract(wo, bx::face_forward(glm::vec3(0, 0, 1), wo), eta, wi)) {
            // total internal reflection
            sample.type = BxdfFlag::Unset;
            sample.f = glm::vec3(0);
            return sample;
        }
        sample.wi = wi;
        sample.pdf = t / (r + t);
        sample.f = eta * eta * glm::vec3(t);
        sample.type = BxdfFlag::Transmission;
    }

    sample.type |= BxdfFlag::Specular;

    return sample;
}

__device__ __forceinline__ BsdfSample sample_dielectric(glm::vec3 p, glm::vec3 wo,
                                                        const Material& m, RngEng& rng) {
    BsdfSample sample{};
    if (m.roughness == 0.f) {
        sample = sample_smooth_dielectric(p, wo, m, rng);
    }

    return sample;
}
