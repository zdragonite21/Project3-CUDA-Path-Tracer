#include "bsdf.cuh"
#include "bxdf_utils.cuh"
#include "conductor.cuh"
#include "dielectric.cuh"
#include "diffuse.cuh"

struct EvalOp {
    glm::vec3 wo, wi;
    __device__ glm::vec3 operator()(const Lambertian& m) const {
        return eval_diffuse(wi, m);
    }
    __device__ glm::vec3 operator()(const Conductor& m) const {
        return eval_conductor(wi, wo, m);
    }
    __device__ glm::vec3 operator()(const Dielectric& m) const {
        return eval_dielectric(wi, wo, m);
    }
};

struct PdfOp {
    glm::vec3 wo, wi;
    __device__ float operator()(const Lambertian& m) const {
        return pdf_diffuse(wi);
    }
    __device__ float operator()(const Conductor& m) const {
        return pdf_conductor(wi, wo, m);
    }
    __device__ float operator()(const Dielectric& m) const {
        return pdf_dielectric(wi, wo, m);
    }
};

struct SampleOp {
    glm::vec3 p;
    glm::vec3 wo;
    RngEng& rng;

    __device__ BsdfSample operator()(const Lambertian& m) const {
        return sample_diffuse(p, m, rng);
    }
    __device__ BsdfSample operator()(const Conductor& m) const {
        return sample_conductor(p, wo, m, rng);
    }
    __device__ BsdfSample operator()(const Dielectric& m) const {
        return sample_dielectric(p, wo, m, rng);
    }
};

__device__ glm::vec3 eval_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, glm::vec3 wi_w,
                               const Material& m) {
    glm::vec3 wo = bx::world_to_local(nor) * wo_w;
    glm::vec3 wi = bx::world_to_local(nor) * wi_w;
    if (bx::cos_theta(wo) == 0.f) {
        return glm::vec3(0);
    }
    return cuda::std::visit(EvalOp{wo, wi}, m.bsdf);
}

__device__ float pdf_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo_w, glm::vec3 wi_w,
                          const Material& m) {
    glm::vec3 wo = bx::world_to_local(nor) * wo_w;
    glm::vec3 wi = bx::world_to_local(nor) * wi_w;
    if (bx::cos_theta(wo) == 0.f) {
        return 0.f;
    }

    return cuda::std::visit(PdfOp{wo, wi}, m.bsdf);
}

__device__ BsdfSample sample_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 wo, glm::mat3 to_world, const Material& m,
                                  RngEng& rng) {
    if (bx::cos_theta(wo) == 0.f) {
        return BsdfSample{};
    }

    BsdfSample sample = cuda::std::visit(SampleOp{p, wo, rng}, m.bsdf);

    sample.wi = to_world * sample.wi;
    return sample;
}
