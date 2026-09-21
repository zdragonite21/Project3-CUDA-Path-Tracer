#include "interactions.h"
#include "sampling.cuh"
#include "utilities.h"

__device__ void scatterRay(PathSegment &pathSegment, glm::vec3 intersect,
                           glm::vec3 normal, const Material &m,
                           thrust::default_random_engine &rng) {
    // sample BSDF
    glm::vec3 wi;
    float pdf;
    glm::vec3 bsdf;
    if (m.type == SPECULAR) {
        glm::vec3 wo = pathSegment.ray.direction;
        wi = glm::reflect(wo, normal);
        pdf = 1.f;
        bsdf = m.color / glm::abs(glm::dot(wi, normal));
    } else {
        wi = calculateRandomDirectionInCosineHemisphere(normal, rng);
        pdf = glm::dot(wi, normal) * INV_PI;
        bsdf = m.color * INV_PI;
    }

    float lambert = glm::abs(glm::dot(wi, normal));

    if (pdf == 0.0) {
        pathSegment.throughput = glm::vec3(0.0);
        pathSegment.remainingBounces = 0;
    } else {
        pathSegment.throughput *= bsdf * lambert / pdf;
        pathSegment.ray = Ray{intersect, wi};
        pathSegment.remainingBounces--;
    }
}
