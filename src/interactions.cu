#include "bxdf_utils.cuh"
#include "interactions.h"
#include "sampling.cuh"
#include "utilities.h"

__device__ glm::vec3 fresnelConductorEval(float cosThetaI,
                                          const glm::vec3 &etaI,
                                          const glm::vec3 &etaT,
                                          const glm::vec3 &k) {
    cosThetaI = glm::clamp(std::abs(cosThetaI), 0.f, 1.f);

    glm::vec3 eta = etaT / etaI;
    glm::vec3 etaK = k / etaI;

    float cos2Theta = cosThetaI * cosThetaI;
    float sin2Theta = 1.f - cos2Theta;
    float sin4Theta = sin2Theta * sin2Theta;

    glm::vec3 eta2 = eta * eta;
    glm::vec3 k2 = etaK * etaK;

    glm::vec3 a2PlusB2 = glm::sqrt(
        (eta2 - k2 - sin2Theta) * (eta2 - k2 - sin2Theta) + 4.f * eta2 * k2);

    glm::vec3 a = glm::sqrt(0.5f * (a2PlusB2 + eta2 - k2 - sin2Theta));

    glm::vec3 rPerp = (a2PlusB2 - 2.f * a * cosThetaI + cos2Theta) /
                      (a2PlusB2 + 2.f * a * cosThetaI + cos2Theta);

    glm::vec3 rParallel =
        rPerp *
        (cos2Theta * a2PlusB2 - 2.f * a * cosThetaI * sin2Theta + sin4Theta) /
        (cos2Theta * a2PlusB2 + 2.f * a * cosThetaI * sin2Theta + sin4Theta);

    return 0.5f * (rPerp + rParallel);
}

__device__ float fresnelDielectricEval(float cosThetaI, float etaI,
                                       float etaT) {
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

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}

__device__ BSDFSample sampleDiffuse(glm::vec3 p, glm::vec3 wo,
                                    const Material &m,
                                    thrust::default_random_engine &rng) {
    BSDFSample sample;
    sample.wi = calculateRandomDirectionInCosineHemisphere(rng);
    sample.pdf = bx::CosTheta(sample.wi) * INV_PI;
    sample.f = m.color * INV_PI;
    sample.type = BxDFFlag::Diffuse;

    return sample;
}

// assumes the medium is air and not spectral
__device__ BSDFSample
sampleSmoothDielectric(glm::vec3 p, glm::vec3 wo, const Material &m,
                         thrust::default_random_engine &rng) {
    BSDFSample sample;
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // etaI = 1.f because we assume air here
    float r = fresnelDielectricEval(bx::CosTheta(wo), 1.0f, m.ior);
    float t = 1.f - r;

    if (u01(rng) < r / (r + t)) {
        // sample perfect specular reflection
        sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
        sample.pdf = r / (r + t);
        sample.f = glm::vec3(r) / glm::abs(bx::CosTheta(sample.wi));
        sample.type = BxDFFlag::Reflection;
    } else {
        // sample perfect specular transmission
        bool entering = bx::CosTheta(wo) > 0;
        float etaI = entering ? 1.0 : m.ior;
        float etaT = entering ? m.ior : 1.0;
        float eta = etaI / etaT;
        glm::vec3 wi;
        if (!bx::Refract(wo, bx::Faceforward(glm::vec3(0, 0, 1), wo),
                         eta, wi)) {
            // total internal reflection
            sample.type = BxDFFlag::Unset;
            sample.f = glm::vec3(0);
            return sample;
        }
        sample.wi = wi;
        sample.pdf = t / (r + t);
        sample.f = eta * eta * glm::vec3(t) / glm::abs(bx::CosTheta(sample.wi));
        sample.type = BxDFFlag::Transmission;
    }

    sample.type |= BxDFFlag::Specular;

    return sample;
}

// assumes the medium is air and rgb approx, not spectral
__device__ BSDFSample sampleSmoothConductor(glm::vec3 p, glm::vec3 wo,
                                              const Material &m) {
    BSDFSample sample;

    sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
    sample.pdf = 1.f;
    // etaI for air is 1
    glm::vec3 fr =
        fresnelConductorEval(bx::CosTheta(sample.wi), glm::vec3(1), m.eta, m.k);
    sample.f = fr / glm::abs(bx::CosTheta(sample.wi));
    sample.type = BxDFFlag::Reflection | BxDFFlag::Specular;

    return sample;
}

__device__ BSDFSample sampleDielectric(glm::vec3 p, glm::vec3 wo,
                                       const Material &m,
                                       thrust::default_random_engine &rng) {
    BSDFSample sample{};
    if (m.roughness == 0.0) {
        sample = sampleSmoothDielectric(p, wo, m, rng);
    }

    return sample;
}

__device__ BSDFSample sampleConductor(glm::vec3 p, glm::vec3 wo,
                                      const Material &m,
                                      thrust::default_random_engine &rng) {
    BSDFSample sample{};
    if (m.roughness == 0.0) {
        sample = sampleSmoothConductor(p, wo, m);
    }

    return sample;
}

__device__ BSDFSample sampleBSDF(glm::vec3 p, glm::vec3 nor, glm::vec3 woW,
                                 const Material &m,
                                 thrust::default_random_engine &rng) {
    glm::vec3 wo = bx::worldToLocal(nor) * woW;

    BSDFSample sample{};
    switch (m.type) {
    case MatType::DIFFUSE:
        sample = sampleDiffuse(p, wo, m, rng);
        break;
    case MatType::DIELECTRIC:
        sample = sampleDielectric(p, wo, m, rng);
        break;
    case MatType::CONDUCTOR:
        sample = sampleConductor(p, wo, m, rng);
        break;
    }

    sample.wi = bx::localToWorld(nor) * sample.wi;
    return sample;
}

__device__ glm::vec3 evalBSDF() { return glm::vec3(0); }

__device__ float pdfBSDF() { return 0.0; }

__device__ void scatterRay(PathSegment &pathSegment, glm::vec3 p,
                           glm::vec3 normal, const Material &m,
                           thrust::default_random_engine &rng) {

    BSDFSample s =
        sampleBSDF(p, normal, -pathSegment.ray.dir, m, rng);

    float lambert = glm::abs(glm::dot(s.wi, normal));

    if (s.type == BxDFFlag::Unset || s.pdf == 0.0) {
        pathSegment.throughput = glm::vec3(0.0);
        pathSegment.remainingBounces = 0;
    } else {
        pathSegment.throughput *= s.f * lambert / s.pdf;
        pathSegment.ray = bx::SpawnRay(p, s.wi);
        pathSegment.remainingBounces--;
    }
}
