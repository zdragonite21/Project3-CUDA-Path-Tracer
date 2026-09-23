#include "interactions.h"
#include "sampling.cuh"
#include "utilities.h"

__device__ void coordinateSystem(glm::vec3 in_nor, glm::vec3 &out_tan,
                                 glm::vec3 &out_bit) {
    if (abs(in_nor.x) > abs(in_nor.y))
        out_tan = glm::vec3(-in_nor.z, 0, in_nor.x) /
                  sqrt(in_nor.x * in_nor.x + in_nor.z * in_nor.z);
    else
        out_tan = glm::vec3(0, in_nor.z, -in_nor.y) /
                  sqrt(in_nor.y * in_nor.y + in_nor.z * in_nor.z);
    out_bit = glm::cross(in_nor, out_tan);
}

__device__ glm::mat3 localToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__device__ glm::mat3 worldToLocal(glm::vec3 nor) {
    return glm::transpose(localToWorld(nor));
}

__device__ BSDFSample sampleDiffuse(glm::vec3 p, glm::vec3 wo,
                                    const Material &m,
                                    thrust::default_random_engine &rng) {
    BSDFSample sample;
    sample.wi = calculateRandomDirectionInCosineHemisphere(rng);
    sample.pdf = sample.wi.z * INV_PI;
    sample.f = m.color * INV_PI;
    sample.type = BxDFFlag::Diffuse;

    return sample;
}

__device__ BSDFSample sampleDielectric(glm::vec3 p, glm::vec3 wo,
                                       const Material &m,
                                       thrust::default_random_engine &rng) {
    BSDFSample sample{};
    sample.type = BxDFFlag::Reflection;

    return sample;
}

__device__ BSDFSample sampleConductor(glm::vec3 p, glm::vec3 wo,
                                      const Material &m,
                                      thrust::default_random_engine &rng) {
    BSDFSample sample;
    sample.wi = glm::reflect(-wo, glm::vec3(0, 0, 1));
    sample.pdf = 1.f;
    sample.f = m.eta / glm::abs(sample.wi.z);
    sample.type = BxDFFlag::Specular;

    return sample;
}

__device__ BSDFSample sampleBSDF(glm::vec3 p, glm::vec3 nor, glm::vec3 woW,
                                 const Material &m,
                                 thrust::default_random_engine &rng) {
    glm::vec3 wo = worldToLocal(nor) * woW;

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

    sample.wi = localToWorld(nor) * sample.wi;
    return sample;
}

__device__ glm::vec3 evalBSDF() { return glm::vec3(0); }

__device__ glm::vec3 pdfBSDF() { return glm::vec3(0); }

__device__ void scatterRay(PathSegment &pathSegment, glm::vec3 intersect,
                           glm::vec3 normal, const Material &m,
                           thrust::default_random_engine &rng) {

    BSDFSample s =
        sampleBSDF(intersect, normal, -pathSegment.ray.direction, m, rng);

    float lambert = glm::abs(glm::dot(s.wi, normal));

    if (s.pdf == 0.0) {
        pathSegment.throughput = glm::vec3(0.0);
        pathSegment.remainingBounces = 0;
    } else {
        pathSegment.throughput *= s.f * lambert / s.pdf;
        pathSegment.ray = Ray{intersect, s.wi};
        pathSegment.remainingBounces--;
    }
}
