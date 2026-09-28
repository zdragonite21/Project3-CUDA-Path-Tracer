#include "intersections.cuh"
#include "light_sampling.cuh"
#include <glm/gtx/norm.hpp>

__device__ float area_to_solid_angle_pdf(float pdf_area, float dist2, float cos_light) {
    return dist2 * pdf_area / cos_light;
}

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

__device__ float pdf_plane_light(Ray r, const Geom& plane) {
    // for mis, assume ray hits light properly
    glm::vec3 hit_pt;
    bool outside;
    float t = plane_intersection_test(plane, r, &hit_pt, nullptr, &outside);

    if (t <= 0.f || outside) {
        return 0.f;
    }

    glm::vec3 light_normal =
        glm::normalize(multiply_mv(plane.transform.inv_transpose, glm::vec4(0, 1, 0, 0)));

    float surface_area = plane.transform.scale.x * plane.transform.scale.z;
    float dist2 = glm::length2(hit_pt - r.org);
    float cos_light = glm::dot(-r.dir, light_normal);
    if (cos_light <= 0.0001) {
        return 0.f;
    }
    return area_to_solid_angle_pdf(1.f / surface_area, dist2, cos_light);
}

__device__ float pdf_li(const Ray& r, const Light& light, const Geom* geoms, int num_geoms) {
    switch (light.type) {
    case LightType::Area:
        return pdf_plane_light(r, geoms[light.geom_id]);
    case LightType::Environment:
        // not supported yet
        return 0.f;
    }
}

// sample one light
__device__ cstd::optional<LightSample> sample_li(glm::vec3 p, glm::vec3 nor, const Light& light,
                                                 const Geom* geoms, int num_geoms, RngEng& rng) {
    switch (light.type) {
    case LightType::Area: {
        const Geom& geom = geoms[light.geom_id];
        switch (geom.type) {
        case GeomType::Plane:
            return sample_plane_light(p, geom, rng);
        default:
            return cstd::nullopt;
        }
    }
    case LightType::Environment:
        // not supported yet
        return cstd::nullopt;
    default:
        return cstd::nullopt;
    }
}

// choose a light to sample
__device__ cstd::optional<LightSample> sample_direct_light(glm::vec3 p, glm::vec3 nor,
                                                           const Light* lights, int num_lights,
                                                           const Geom* geoms, int num_geoms,
                                                           RngEng& rng) {
    if (num_lights == 0) {
        return cstd::nullopt;
    }
    UnifDist<float> u01(0, 1);

    int light_idx = static_cast<int>(u01(rng) * num_lights);
    const Light& light = lights[light_idx];

    cstd::optional<LightSample> sample = sample_li(p, nor, light, geoms, num_geoms, rng);

    if (!sample || glm::dot(sample->wi, nor) <= 0.f) {
        return cstd::nullopt;
    }

    sample->pdf /= num_lights;
    sample->light_idx = light_idx;

    return sample;
}
