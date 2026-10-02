#include "bsdf/bsdf.cuh"
#include "bsdf/bxdf_utils.cuh"
#include "config.h"
#include "intersections.cuh"
#include "light_sampling.cuh"
#include "sampling.cuh"
#include "scene_structs.h"
#include "shading.cuh"
#include "math_utils.h"
#include <corecrt_terminate.h>
#include "render_settings.cuh"

__device__ __forceinline__ float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2);
}

__device__ __forceinline__ void terminate_path(PathSegment& path) {
    path.remaining_bounces = 0;
}

__device__ glm::vec3 eval_environment(const DeviceEnvMap& env, glm::vec3 wi) {
    if (env.texture == 0) {
        return glm::vec3(0.0f);
    }

    glm::vec3 d = glm::normalize(wi);

    float theta = acosf(glm::clamp(d.y, -1.f, 1.f));
    float phi = atan2f(d.z, d.x);
    if (phi < 0.f) {
        phi += 2.f * PI;
    }

    float u = phi / (2.f * PI);
    float v = theta / PI;

    float4 rgb = tex2D<float4>(env.texture, u, v);
    return c_settings.env_strength * glm::vec3(rgb.x, rgb.y, rgb.z);
}

__device__ cstd::optional<ShadowRay>
estimate_direct_lighting(const PathSegment& path, glm::vec3 p, glm::vec3 nor, const Material& m,
                         RngEng& rng, const Light* lights, int num_lights, const Geom* geoms,
                         int num_geoms, const DeviceEnvMap& env) {
    cstd::optional<LightSample> ls =
        sample_direct_light(p, nor, lights, num_lights, geoms, num_geoms, rng);

    if (!ls || ls->pdf == 0.f) {
        return cstd::nullopt;
    }

    const Light& light = lights[ls->light_idx];

    glm::vec3 li =
        light.type == LightType::Environment ? eval_environment(env, ls->wi) : light.emission;

    glm::vec3 bsdf_f = eval_bsdf(p, nor, -path.ray.dir, ls->wi, m);
    float bsdf_p = pdf_bsdf(p, nor, -path.ray.dir, ls->wi, m);

    float w = path.remaining_bounces > 1 ? power_heuristic(ls->pdf, bsdf_p) : 1.f;

    ShadowRay sray{};
    sray.ray = bx::spawn_ray(p, ls->wi, nor);
    sray.pixel_index = path.pixel_index;
    // so we don't intersect with the same light when tracing shadow rays
    sray.t_max = light.type == LightType::Environment ? FLT_MAX : ls->dist - numeric::shadow_margin;

    sray.contribution = path.throughput * li * bsdf_f * w / ls->pdf;

    return sray;
}

__device__ void scatter_ray(PathSegment& path_segment, glm::vec3 p, glm::vec3 normal,
                            const Material& m, RngEng& rng) {

    BsdfSample bs = sample_bsdf(p, normal, -path_segment.ray.dir, m, rng);

    if (bs.type == BxdfFlag::Unset || bs.pdf == 0.f) {
        path_segment.throughput = glm::vec3(0.f);
        terminate_path(path_segment);
    } else {
        path_segment.throughput *= bs.f / bs.pdf;
        path_segment.prev_was_delta = (bs.type & BxdfFlag::Specular) != BxdfFlag::Unset;
        path_segment.prev_bsdf_pdf = bs.pdf;
        path_segment.ray = bx::spawn_ray(p, bs.wi, normal);
        path_segment.remaining_bounces--;
    }
}

__device__ __forceinline__ bool is_not_specular(const Material& m) {
    return m.type == MatType::Diffuse ||
           ((m.type == MatType::Dielectric || m.type == MatType::Conductor) && m.roughness != 0.f);
}

__global__ void shade_material(int iter, int num_paths, int depth, int num_lights, int num_geoms,
                               const ShadeableIntersection* shadeable_intersections,
                               const MatId* isect_mat_ids, PathSegment* path_segments,
                               ShadowRay* shadow_rays, const Material* materials,
                               const Light* lights, const Geom* geoms, glm::vec3* image,
                               DeviceEnvMap env) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths || path_segments[idx].remaining_bounces <= 0) {
        return;
    }

    shadow_rays[idx].pixel_index = -1;

    // miss / environment

    PathSegment path = path_segments[idx];
    MatId mat_id = isect_mat_ids[idx];
    if (mat_id == UINT8_MAX) {
        glm::vec3 le = eval_environment(env, path.ray.dir);
        float w = 1.f;
        
#if LI_MIS
        if (depth > 0 && !path.prev_was_delta && env.light_idx >= 0) {
            float li_pdf = pdf_li(path.ray, lights[env.light_idx], geoms, num_geoms) /
                           static_cast<float>(num_lights);
            w = power_heuristic(path.prev_bsdf_pdf, li_pdf);
        }
#endif
        image[path.pixel_index] += path.throughput * le * w;
        terminate_path(path_segments[idx]);
        return;
    }

    // emissive

    Material material = materials[mat_id];
    if (material.type == MatType::Emissive && material.emission != glm::vec3(0)) {

#if LI_MIS
        if (depth == 0 || path.prev_was_delta) {
            image[path.pixel_index] += path.throughput * material.emission;
        } else {
            // light sampling
            float li_pdf =
                pdf_li(path.ray, lights[shadeable_intersections[idx].light_idx], geoms, num_geoms) /
                static_cast<float>(num_lights);

            float w = power_heuristic(path.prev_bsdf_pdf, li_pdf);
            image[path.pixel_index] += w * path.throughput * material.emission;
        }
#else
        image[path.pixel_index] += path.throughput * material.emission;
#endif

        terminate_path(path_segments[idx]);
        return;
    }

    // bounce

    ShadeableIntersection intersection = shadeable_intersections[idx];
    RngEng rng = make_seeded_rng(iter, idx, depth);
    glm::vec3 p = get_point_on_ray(path.ray, intersection.t);
    const glm::vec3& nor = intersection.surface_normal;

#if LI_MIS
    if (is_not_specular(material)) {
        // if not delta (so change this when I add microfacet)
        cstd::optional<ShadowRay> sray = estimate_direct_lighting(
            path, p, nor, material, rng, lights, num_lights, geoms, num_geoms, env);

        if (sray) {
            shadow_rays[idx] = *sray;
        }
    }

    scatter_ray(path, p, nor, material, rng);
#else
    scatter_ray(path, p, nor, material, rng);
#endif

#if RUSSIAN_ROULETTE
    if (depth > 3 && path.remaining_bounces > 0) {
        UnifDist<float> u01(0, 1);

        float survive_p = max(path.throughput.x, max(path.throughput.y, path.throughput.z));
        survive_p = glm::clamp(survive_p, 0.05f, 0.95f);
        if (u01(rng) > survive_p) {
            terminate_path(path_segments[idx]);
            return;
        } else {
            path.throughput /= survive_p;
        }
    }
#endif
    path_segments[idx] = path;
}

__global__ void trace_shadow_rays(int num_srays, int num_geoms, ShadowRay* shadow_rays,
                                  const Geom* geoms, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_srays || shadow_rays[idx].pixel_index == -1) {
        return;
    }

    ShadowRay sray = shadow_rays[idx];
    if (visible_to_light(sray.ray, sray.t_max, geoms, num_geoms)) {
        image[sray.pixel_index] += sray.contribution;
    }
}
