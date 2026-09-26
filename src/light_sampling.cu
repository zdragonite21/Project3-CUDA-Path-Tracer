#include "bxdf_utils.cuh"
#include "intersections.h"
#include "light_sampling.cuh"
#include "sampling.cuh"
#include "sceneStructs.h"

__device__ cstd::optional<LightSample> directSamplePlaneLight(glm::vec3 p, const Geom& plane,
                                                             RngEng& rng) {
    LightSample sample{};
    UnifDist<float> u01(0, 1);

    float surfaceArea = plane.transform.scale.x * plane.transform.scale.z;
    glm::vec2 xi(u01(rng) - 0.5f, u01(rng) - 0.5f);
    glm::vec3 lightP = multiplyMV(plane.transform.matrix, glm::vec4(xi.x, 0, xi.y, 1));
    glm::vec3 lightN =
        glm::normalize(multiplyMV(plane.transform.invTranspose, glm::vec4(0, 1, 0, 0)));

    glm::vec3 v = lightP - p;
    sample.dist = length(v);
    if (sample.dist == 0.f) {
        return cstd::nullopt;
    }
    sample.wi = v / sample.dist;

    float cosT = glm::dot(-sample.wi, lightN);
    if (cosT <= 0.0001) {
        return cstd::nullopt;
    }

    sample.pdf = sample.dist * sample.dist / (cosT * surfaceArea);
    return sample;
}

__device__ cstd::optional<LightSample> directSampleAreaLight(glm::vec3 p, const Geom& geom,
                                                            RngEng& rng) {
    switch (geom.type) {
    case GeomType::PLANE:
        return directSamplePlaneLight(p, geom, rng);
    default:
        return cstd::nullopt;
    }
}

__device__ cstd::optional<LightSample> sampleLi(glm::vec3 p, glm::vec3 nor, const Light* lights,
                                               int lights_size, const Geom* geoms, int geoms_size,
                                               RngEng& rng) {
    if (lights_size == 0) {
        return cstd::nullopt;
    }
    UnifDist<float> u01(0, 1);

    int lightIdx = static_cast<int>(u01(rng) * lights_size);
    const Light& light = lights[lightIdx];

    cstd::optional<LightSample> sample;
    
    switch (light.type) {
    case LightType::AREA:
        sample = directSampleAreaLight(p, geoms[light.geomId], rng);
        break;
    case LightType::ENVIRONMENT:
        // not supported yet
        return cstd::nullopt;
    }
    if (!sample || glm::dot(sample->wi, nor) <= 0.f) {
        return cstd::nullopt;
    }

    Ray shadowRay = bx::SpawnRay(p, sample->wi);
    if (visibleToLight(shadowRay, light.geomId, sample->dist, geoms, geoms_size)) {
        sample->pdf /= lights_size;
        sample->radiance = light.emission;
        sample->lightIdx = lightIdx;
        return *sample;
    } else {
        return cstd::nullopt;
    }
}
