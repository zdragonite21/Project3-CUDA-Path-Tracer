#include "bsdf.cuh"
#include "bxdf_utils.cuh"
#include "sampling.cuh"
#include "utilities.h"

#include <cmath>

__device__ glm::vec3 fresnel_conductor_eval(float cos_theta_i, const glm::vec3& eta_i,
                                          const glm::vec3& eta_t, const glm::vec3& k) {
    cos_theta_i = glm::clamp(std::abs(cos_theta_i), 0.f, 1.f);

    glm::vec3 eta = eta_t / eta_i;
    glm::vec3 eta_k = k / eta_i;

    float cos2_theta = cos_theta_i * cos_theta_i;
    float sin2_theta = 1.f - cos2_theta;
    float sin4_theta = sin2_theta * sin2_theta;

    glm::vec3 eta2 = eta * eta;
    glm::vec3 k2 = eta_k * eta_k;

    glm::vec3 a2_plus_b2 =
        glm::sqrt((eta2 - k2 - sin2_theta) * (eta2 - k2 - sin2_theta) + 4.f * eta2 * k2);

    glm::vec3 a = glm::sqrt(0.5f * (a2_plus_b2 + eta2 - k2 - sin2_theta));

    glm::vec3 r_perp =
        (a2_plus_b2 - 2.f * a * cos_theta_i + cos2_theta) / (a2_plus_b2 + 2.f * a * cos_theta_i + cos2_theta);

    glm::vec3 r_parallel = r_perp *
                          (cos2_theta * a2_plus_b2 - 2.f * a * cos_theta_i * sin2_theta + sin4_theta) /
                          (cos2_theta * a2_plus_b2 + 2.f * a * cos_theta_i * sin2_theta + sin4_theta);

    return 0.5f * (r_perp + r_parallel);
}

__device__ float fresnel_dielectric_eval(float cos_theta_i, float eta_i, float eta_t) {
    cos_theta_i = glm::clamp(cos_theta_i, -1.f, 1.f);
    bool entering = cos_theta_i > 0.f;
    if (!entering) {
        float tmp = eta_i;
        eta_i = eta_t;
        eta_t = tmp;
        cos_theta_i = std::abs(cos_theta_i);
    }

    float sin_theta_i = glm::sqrt(glm::max(0.f, 1 - cos_theta_i * cos_theta_i));
    float sin_theta_t = eta_i / eta_t * sin_theta_i;
    if (sin_theta_t >= 1.f) {
        // total internal reflection
        return 1.f;
    }
    float cos_theta_t = glm::sqrt(glm::max(0.f, 1.f - sin_theta_t * sin_theta_t));

    float r_parl =
        ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    float r_perp =
        ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parl * r_parl + r_perp * r_perp) / 2.f;
}

__device__ __forceinline__ glm::vec3 eval_diffuse(glm::vec3 color) {
    return color * INV_PI;
}

__device__ BsdfSample sample_diffuse(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample;
    sample.wi = calculate_random_direction_in_cosine_hemisphere(rng);
    sample.pdf = bx::cos_theta(sample.wi) * INV_PI;
    sample.f = eval_diffuse(m.color);
    sample.type = BxdfFlag::Diffuse;

    return sample;
}

// assumes the medium is air and not spectral
__device__ BsdfSample sample_smooth_dielectric(glm::vec3 p, glm::vec3 wo, const Material& m,
                                             RngEng& rng) {
    BsdfSample sample;
    UnifDist<float> u01(0, 1);

    // eta_i = 1.f because we assume air here
    float r = fresnel_dielectric_eval(bx::cos_theta(wo), 1.0f, m.ior);
    float t = 1.f - r;

    if (u01(rng) < r / (r + t)) {
        // sample perfect specular reflection
        sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
        sample.pdf = r / (r + t);
        sample.f = glm::vec3(r) / glm::abs(bx::cos_theta(sample.wi));
        sample.type = BxdfFlag::Reflection;
    } else {
        // sample perfect specular transmission
        bool entering = bx::cos_theta(wo) > 0;
        float eta_i = entering ? 1.0 : m.ior;
        float eta_t = entering ? m.ior : 1.0;
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
        sample.f = eta * eta * glm::vec3(t) / glm::abs(bx::cos_theta(sample.wi));
        sample.type = BxdfFlag::Transmission;
    }

    sample.type |= BxdfFlag::Specular;

    return sample;
}

// assumes the medium is air and rgb approx, not spectral
__device__ BsdfSample sample_smooth_conductor(glm::vec3 p, glm::vec3 wo, const Material& m) {
    BsdfSample sample;

    sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
    sample.pdf = 1.f;
    // eta_i for air is 1
    glm::vec3 fr = fresnel_conductor_eval(bx::cos_theta(sample.wi), glm::vec3(1), m.eta, m.k);
    sample.f = fr / glm::abs(bx::cos_theta(sample.wi));
    sample.type = BxdfFlag::Reflection | BxdfFlag::Specular;

    return sample;
}

__device__ BsdfSample sample_dielectric(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample{};
    if (m.roughness == 0.0) {
        sample = sample_smooth_dielectric(p, wo, m, rng);
    }

    return sample;
}

__device__ BsdfSample sample_conductor(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample{};
    if (m.roughness == 0.0) {
        sample = sample_smooth_conductor(p, wo, m);
    }

    return sample;
}

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
    if (wo.z == 0.0) {
        return glm::vec3(0);
    }

    switch (m.type) {
    case MatType::Diffuse:
        return eval_diffuse(m.color);
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

__device__ float pdf_bsdf() {
    return 0.0;
}

__device__ void scatter_ray(PathSegment& path_segment, glm::vec3 p, glm::vec3 normal,
                           const Material& m, RngEng& rng) {

    BsdfSample s = sample_bsdf(p, normal, -path_segment.ray.dir, m, rng);

    float lambert = glm::abs(glm::dot(s.wi, normal));

    if (s.type == BxdfFlag::Unset || s.pdf == 0.0) {
        path_segment.throughput = glm::vec3(0.0);
        path_segment.remaining_bounces = 0;
    } else {
        path_segment.throughput *= s.f * lambert / s.pdf;
        path_segment.ray = bx::spawn_ray(p, s.wi);
        path_segment.remaining_bounces--;
    }
}
