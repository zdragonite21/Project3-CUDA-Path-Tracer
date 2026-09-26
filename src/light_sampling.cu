#include "bxdf_utils.cuh"
#include "intersections.h"
#include "light_sampling.cuh"
#include "sampling.cuh"
#include "scene_structs.h"

__device__ cstd::optional<LightSample> sample_plane_light(glm::vec3 p, const Geom& plane,
                                                             RngEng& rng) {
    LightSample sample{};
    UnifDist<float> u01(0, 1);

    float surface_area = plane.transform.scale.x * plane.transform.scale.z;
    glm::vec2 xi(u01(rng) - 0.5f, u01(rng) - 0.5f);
    glm::vec3 light_pos = multiply_mv(plane.transform.matrix, glm::vec4(xi.x, 0, xi.y, 1));
    glm::vec3 light_normal =
        glm::normalize(multiply_mv(plane.transform.inv_transpose, glm::vec4(0, 1, 0, 0)));

    glm::vec3 v = light_pos - p;
    sample.dist = length(v);
    if (sample.dist == 0.f) {
        return cstd::nullopt;
    }
    sample.wi = v / sample.dist;

    float cos_t = glm::dot(-sample.wi, light_normal);
    if (cos_t <= 0.0001) {
        return cstd::nullopt;
    }

    sample.pdf = sample.dist * sample.dist / (cos_t * surface_area);
    return sample;
}

__device__ cstd::optional<LightSample> sample_area_light(glm::vec3 p, const Geom& geom,
                                                            RngEng& rng) {
    switch (geom.type) {
    case GeomType::Plane:
        return sample_plane_light(p, geom, rng);
    default:
        return cstd::nullopt;
    }
}

__device__ cstd::optional<LightSample> sample_li(glm::vec3 p, glm::vec3 nor, const Light* lights,
                                               int num_lights, const Geom* geoms, int num_geoms,
                                               RngEng& rng) {
    if (num_lights == 0) {
        return cstd::nullopt;
    }
    UnifDist<float> u01(0, 1);

    int light_idx = static_cast<int>(u01(rng) * num_lights);
    const Light& light = lights[light_idx];

    cstd::optional<LightSample> sample;
    
    switch (light.type) {
    case LightType::Area:
        sample = sample_area_light(p, geoms[light.geom_id], rng);
        break;
    case LightType::Environment:
        // not supported yet
        return cstd::nullopt;
    }
    if (!sample || glm::dot(sample->wi, nor) <= 0.f) {
        return cstd::nullopt;
    }

    Ray shadow_ray = bx::spawn_ray(p, sample->wi);
    if (visible_to_light(shadow_ray, light.geom_id, sample->dist, geoms, num_geoms)) {
        sample->pdf /= num_lights;
        sample->radiance = light.emission;
        sample->light_idx = light_idx;
        return *sample;
    } else {
        return cstd::nullopt;
    }
}
