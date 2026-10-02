#include "bsdf.cuh"
#include "bxdf_utils.cuh"
#include "dielectric.cuh"
#include "conductor.cuh"
#include "diffuse.cuh"

__device__ BsdfSample sample_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, const Material& m,
                                  RngEng& rng) {
    glm::vec3 wo = bx::world_to_local(nor) * wo_w;

    BsdfSample sample{};
    switch (m.type) {
    case MatType::Diffuse:
        sample = sample_diffuse(p, wo, m, rng);
        break;
    case MatType::Dielectric:
        sample = sample_dielectric(p, wo, m, rng);
        break;
    case MatType::Conductor:
        sample = sample_conductor(p, wo, m, rng);
        break;
    case MatType::Emissive:
        return sample;
    }

    sample.wi = bx::local_to_world(nor) * sample.wi;
    return sample;
}

__device__ glm::vec3 eval_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, glm::vec3 wi_w,
                               const Material& m) {
    glm::vec3 wo = bx::world_to_local(nor) * wo_w;
    glm::vec3 wi = bx::world_to_local(nor) * wi_w;

    // lambertian term will be 0
    if (wo.z == 0.f) {
        return glm::vec3(0);
    }

    switch (m.type) {
    case MatType::Diffuse:
        return eval_diffuse(wi, m.color);
    case MatType::Dielectric:
        if (m.roughness == 0.f) {
            return glm::vec3(0.f);
        }
        // implement microfacet
        return glm::vec3(0.f);
    case MatType::Conductor:
        if (m.roughness == 0.f) {
            return glm::vec3(0.f);
        }
        // implement microfacet
        return glm::vec3(0.f);
    default:
        return glm::vec3(0.f);
    }
}

__device__ float pdf_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, glm::vec3 wi_w,
                          const Material& m) {
    glm::vec3 wo = bx::world_to_local(nor) * wo_w;
    glm::vec3 wi = bx::world_to_local(nor) * wi_w;

    if (wo.z == 0.f) {
        return 0.f;
    }

    switch (m.type) {
    case MatType::Diffuse:
        return pdf_diffuse(wi);
    case MatType::Dielectric:
        if (m.roughness == 0.f) {
            return 0.f;
        }
        // implement microfacet
        return 0.f;
    case MatType::Conductor:
        if (m.roughness == 0.f) {
            return 0.f;
        }
        // implement microfacet
        return 0.f;
    default:
        return 0.f;
    }
}
