#include "bxdf_utils.cuh"
#include "intersections.h"
#include "light_sampling.cuh"
#include "sampling.cuh"
#include "sceneStructs.h"

__device__ LightSample directSamplePlaneLight(glm::vec3 p, const Geom& plane, RngEng& rng) {
    LightSample sample{};
    UnifDist<float> u01(0, 1);

    float surfaceArea = plane.transform.scale.x * plane.transform.scale.z;
    glm::vec2 xi(u01(rng) - 0.5f, u01(rng) - 0.5f);
    glm::vec3 lightP = multiplyMV(plane.transform.matrix, glm::vec4(xi.x, 0, xi.y, 1));
    glm::vec3 lightN =
        glm::normalize(multiplyMV(plane.transform.invTranspose, glm::vec4(0, 1, 0, 0)));

    glm::vec3 v = lightP - p;
    sample.dist = length(v);
    sample.wi = v / sample.dist;

    float cosT = glm::dot(-sample.wi, lightN);
    if (cosT <= 0.0001) {
        return sample;
    }

    sample.pdf = sample.dist * sample.dist / (cosT * surfaceArea);
    sample.lightIdx = 0;
    return sample;
}

__device__ LightSample directSampleAreaLight(glm::vec3 p, const Geom& geom, RngEng& rng) {
    LightSample sample{};
    switch (geom.type) {
    case GeomType::PLANE:
        sample = directSamplePlaneLight(p, geom, rng);
        break;
    default:
        break;
    }
    return sample;
}

__device__ LightSample sampleLi(glm::vec3 p, const Light* lights, int lights_size,
                                const Geom* geoms, int geoms_size, RngEng& rng) {
    LightSample sample{};
    if (lights_size == 0) {
        return sample;
    }
    UnifDist<float> u01(0, 1);

    int lightIdx = static_cast<int>(u01(rng) * lights_size);
    const Light& light = lights[lightIdx];

    switch (light.type) {
    case LightType::AREA:
        sample = directSampleAreaLight(p, geoms[light.geomId], rng);
        break;
    case LightType::ENVIRONMENT:
        sample.lightIdx = -1;
        break;
    }
    if (sample.lightIdx == -1) {
        return sample;
    }

    Ray shadowRay = bx::SpawnRay(p, sample.wi);
    if (visibleToLight(shadowRay, light.geomId, sample.dist, geoms, geoms_size)) {
        sample.pdf /= lights_size;
        sample.radiance = light.emission;
        sample.lightIdx = lightIdx;
    }

    return sample;
}
