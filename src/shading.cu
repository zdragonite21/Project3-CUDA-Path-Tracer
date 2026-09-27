#include "bsdf.cuh"
#include "bxdf_utils.cuh"
#include "config.h"
#include "intersections.cuh"
#include "light_sampling.cuh"
#include "sampling.cuh"
#include "scene_structs.h"
#include "shading.cuh"

__device__ inline float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

__device__ void termiante_path(PathSegment& path) {
    path.remaining_bounces = 0;
}

__device__ cstd::optional<ShadowRay> estimate_direct_lighting(const PathSegment& path, glm::vec3 p,
                                                              glm::vec3 nor, const Material& m,
                                                              RngEng& rng, const Light* lights,
                                                              int num_lights, const Geom* geoms,
                                                              int num_geoms) {
    cstd::optional<LightSample> ls = sample_li(p, nor, lights, num_lights, geoms, num_geoms, rng);

    if (!ls || ls->pdf == 0.f) {
        return cstd::nullopt;
    }

    glm::vec3 bsdf_f = eval_bsdf(p, nor, -path.ray.dir, ls->wi, m);
    float bsdf_p = pdf_bsdf(p, nor, -path.ray.dir, ls->wi, m);

    float w = power_heuristic(ls->pdf, bsdf_p);

    ShadowRay sray{};
    sray.ray = bx::spawn_ray(p, ls->wi);
    sray.pixel_index = path.pixel_index;
    sray.t_max = ls->dist;

    float lambert = glm::max(0.f, glm::dot(ls->wi, nor));
    sray.throughput = lights[ls->light_idx].emission * bsdf_f * lambert * w / ls->pdf;

    return sray;
}

__device__ void scatter_ray(PathSegment& path_segment, glm::vec3 p, glm::vec3 normal,
                            const Material& m, RngEng& rng) {

    BsdfSample bs = sample_bsdf(p, normal, -path_segment.ray.dir, m, rng);

    if (bs.type == BxdfFlag::Unset || bs.pdf == 0.0) {
        path_segment.throughput = glm::vec3(0.0);
        termiante_path(path_segment);
    } else {
        path_segment.throughput *= bs.f * bx::abs_dot(bs.wi, normal) / bs.pdf;
        path_segment.ray = bx::spawn_ray(p, bs.wi);
        path_segment.remaining_bounces--;
    }
}

__global__ void shade_material(int iter, int num_paths, int depth, int num_lights, int num_geoms,
                               const ShadeableIntersection* shadeable_intersections,
                               const MatId* isect_mat_ids, PathSegment* path_segments,
                               ShadowRay* shadow_rays, const Material* materials,
                               const Light* lights, const Geom* geoms, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths || path_segments[idx].remaining_bounces <= 0) {
        return;
    }

    // miss / environment

    PathSegment path = path_segments[idx];
    MatId mat_id = isect_mat_ids[idx];
    if (mat_id == UINT8_MAX) {
        // hit env map
        path.throughput = glm::vec3(0.0f);
        path.remaining_bounces = 0;
        path_segments[idx] = path;
        return;
    }

    // emissive

    Material material = materials[mat_id];
    if (material.type == MatType::Emissive) {
        path.remaining_bounces = 0;
        path_segments[idx] = path;
        image[path.pixel_index] += path.throughput * material.emission;
        return;
    }

    // bounce

    glm::vec3 radiance(0);
    ShadeableIntersection intersection = shadeable_intersections[idx];
    RngEng rng = make_seeded_rng(iter, idx, depth);

#if LI_MIS
    glm::vec3 p = get_point_on_ray(path.ray, intersection.t);
    glm::vec3 nor = intersection.surface_normal;

    if (material.type == MatType::Diffuse) {
        // if not delta (so change this when I add microfacet)
        cstd::optional<ShadowRay> sray = estimate_direct_lighting(
            path, p, nor, material, rng, lights, num_lights, geoms, num_geoms);

        if (!sray) {
            shadow_rays[idx].pixel_index = -1;
        } else {
            shadow_rays[idx] = *sray;
        }
    }

    scatter_ray(path, p, nor, material, rng);

#elif LI_DIRECT
    glm::vec3 p = get_point_on_ray(path.ray, intersection.t);
    glm::vec3 direct = estimate_direct_lighting(path, p, intersection.surface_normal, material, rng,
                                                lights, num_lights, geoms, num_geoms);
    radiance += direct;
    path.remaining_bounces = 0;
#else
    glm::vec3 intersect = get_point_on_ray(path.ray, intersection.t);
    scatter_ray(path, intersect, intersection.surface_normal, material, rng);
#endif

#if RUSSIAN_ROULETTE
    if (depth > 3 && path.remaining_bounces > 0) {
        UnifDist<float> u01(0, 1);

        float survive_p = max(path.throughput.x, max(path.throughput.y, path.throughput.z));
        survive_p = glm::clamp(survive_p, 0.05f, 0.95f);
        if (u01(rng) > survive_p) {
            path.throughput = glm::vec3(0);
            path.remaining_bounces = 0;
        } else {
            path.throughput /= survive_p;
        }
    }
#endif
    path_segments[idx] = path;
    // final gather
    image[path.pixel_index] += radiance;
}

__global__ void trace_shadow_rays(int iter, int num_paths, int depth, int num_lights, int num_geoms,
                                  const ShadeableIntersection* shadeable_intersections,
                                  const MatId* isect_mat_ids, PathSegment* path_segments,
                                  ShadowRay* shadow_rays, const Material* materials,
                                  const Light* lights, const Geom* geoms, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths || path_segments[idx].remaining_bounces <= 0) {
        return;
    }

    ShadowRay sray = shadow_rays[idx];

    if (visible_to_light(sray.ray, sray.light_idx, sray.t_max, geoms, num_geoms)) {
        path_segments[idx].throughput = sray.throughput;
    }
}
