#include "bxdf_utils.cuh"
#include "interactions.h"
#include "sampling.cuh"
#include "scene_structs.h"
#include "utilities.h"
#include "light_sampling.cuh"

__device__ glm::vec3 fresnelConductorEval(float cosThetaI, const glm::vec3& etaI,
                                          const glm::vec3& etaT, const glm::vec3& k) {
    cosThetaI = glm::clamp(std::abs(cosThetaI), 0.f, 1.f);

    glm::vec3 eta = etaT / etaI;
    glm::vec3 etaK = k / etaI;

    float cos2Theta = cosThetaI * cosThetaI;
    float sin2Theta = 1.f - cos2Theta;
    float sin4Theta = sin2Theta * sin2Theta;

    glm::vec3 eta2 = eta * eta;
    glm::vec3 k2 = etaK * etaK;

    glm::vec3 a2PlusB2 =
        glm::sqrt((eta2 - k2 - sin2Theta) * (eta2 - k2 - sin2Theta) + 4.f * eta2 * k2);

    glm::vec3 a = glm::sqrt(0.5f * (a2PlusB2 + eta2 - k2 - sin2Theta));

    glm::vec3 rPerp =
        (a2PlusB2 - 2.f * a * cosThetaI + cos2Theta) / (a2PlusB2 + 2.f * a * cosThetaI + cos2Theta);

    glm::vec3 rParallel = rPerp *
                          (cos2Theta * a2PlusB2 - 2.f * a * cosThetaI * sin2Theta + sin4Theta) /
                          (cos2Theta * a2PlusB2 + 2.f * a * cosThetaI * sin2Theta + sin4Theta);

    return 0.5f * (rPerp + rParallel);
}

__device__ float fresnelDielectricEval(float cosThetaI, float etaI, float etaT) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = std::abs(cosThetaI);
    }

    float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.f) {
        // total internal reflection
        return 1.f;
    }
    float costhetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));
    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));

    float Rparl =
        ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp =
        ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}

__device__ __forceinline__ glm::vec3 evalDiffuse(glm::vec3 color) {
    return color * INV_PI;
}

__device__ BsdfSample sampleDiffuse(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample;
    sample.wi = calculateRandomDirectionInCosineHemisphere(rng);
    sample.pdf = bx::cos_theta(sample.wi) * INV_PI;
    sample.f = evalDiffuse(m.color);
    sample.type = BxdfFlag::Diffuse;

    return sample;
}

// assumes the medium is air and not spectral
__device__ BsdfSample sampleSmoothDielectric(glm::vec3 p, glm::vec3 wo, const Material& m,
                                             RngEng& rng) {
    BsdfSample sample;
    UnifDist<float> u01(0, 1);

    // etaI = 1.f because we assume air here
    float r = fresnelDielectricEval(bx::cos_theta(wo), 1.0f, m.ior);
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
        float etaI = entering ? 1.0 : m.ior;
        float etaT = entering ? m.ior : 1.0;
        float eta = etaI / etaT;
        glm::vec3 wi;
        if (!bx::Refract(wo, bx::Faceforward(glm::vec3(0, 0, 1), wo), eta, wi)) {
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
__device__ BsdfSample sampleSmoothConductor(glm::vec3 p, glm::vec3 wo, const Material& m) {
    BsdfSample sample;

    sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
    sample.pdf = 1.f;
    // etaI for air is 1
    glm::vec3 fr = fresnelConductorEval(bx::cos_theta(sample.wi), glm::vec3(1), m.eta, m.k);
    sample.f = fr / glm::abs(bx::cos_theta(sample.wi));
    sample.type = BxdfFlag::Reflection | BxdfFlag::Specular;

    return sample;
}

__device__ BsdfSample sampleDielectric(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample{};
    if (m.roughness == 0.0) {
        sample = sampleSmoothDielectric(p, wo, m, rng);
    }

    return sample;
}

__device__ BsdfSample sampleConductor(glm::vec3 p, glm::vec3 wo, const Material& m, RngEng& rng) {
    BsdfSample sample{};
    if (m.roughness == 0.0) {
        sample = sampleSmoothConductor(p, wo, m);
    }

    return sample;
}

__device__ BsdfSample sample_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 woW, const Material& m,
                                 RngEng& rng) {
    glm::vec3 wo = bx::world_to_local(nor) * woW;

    BsdfSample sample{};
    switch (m.type) {
    case MatType::Diffuse:
        sample = sampleDiffuse(p, wo, m, rng);
        break;
    case MatType::Dielectric:
        sample = sampleDielectric(p, wo, m, rng);
        break;
    case MatType::Conductor:
        sample = sampleConductor(p, wo, m, rng);
        break;
    case MatType::Emissive:
        return sample;
    }

    sample.wi = bx::local_to_world(nor) * sample.wi;
    return sample;
}

__device__ glm::vec3 eval_bsdf(glm::vec3 p, glm::vec3 nor, glm::vec3 woW, glm::vec3 wiW,
                              const Material& m) {
    glm::vec3 wo = bx::world_to_local(nor) * woW;
    glm::vec3 wi = bx::world_to_local(nor) * wiW;

    // lambertian term will be 0
    if (wo.z == 0.0) {
        return glm::vec3(0);
    }

    switch (m.type) {
    case MatType::Diffuse:
        return evalDiffuse(m.color);
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

__device__ void scatter_ray(PathSegment& pathSegment, glm::vec3 p, glm::vec3 normal,
                           const Material& m, RngEng& rng) {

    BsdfSample s = sample_bsdf(p, normal, -pathSegment.ray.dir, m, rng);

    float lambert = glm::abs(glm::dot(s.wi, normal));

    if (s.type == BxdfFlag::Unset || s.pdf == 0.0) {
        pathSegment.throughput = glm::vec3(0.0);
        pathSegment.remaining_bounces = 0;
    } else {
        pathSegment.throughput *= s.f * lambert / s.pdf;
        pathSegment.ray = bx::spawn_ray(p, s.wi);
        pathSegment.remaining_bounces--;
    }
}

__device__ glm::vec3 estimate_direct_lighting(PathSegment& path, glm::vec3 p, glm::vec3 nor, const Material& m,
                          RngEng& rng, const Light* lights, int lights_size, const Geom* geoms,
                          int geoms_size) {
    cstd::optional<LightSample> sample = sample_li(p, nor, lights, lights_size, geoms, geoms_size, rng);

    if (!sample || sample->pdf == 0.f) {
        return glm::vec3(0.f);
    }

    glm::vec3 bsdf = eval_bsdf(p, nor, -path.ray.dir, sample->wi, m);
    float lambert = glm::max(0.f, glm::dot(sample->wi, nor));

    return sample->radiance * bsdf * lambert / sample->pdf;
}
