#include "shading.cuh"
#include "bsdf.cuh"
#include "config.h"
#include "intersections.cuh"
#include "light_sampling.cuh"
#include "sampling.cuh"

__device__ glm::vec3 estimate_direct_lighting(PathSegment& path, glm::vec3 p, glm::vec3 nor, const Material& m,
                          RngEng& rng, const Light* lights, int num_lights, const Geom* geoms,
                          int num_geoms) {
    cstd::optional<LightSample> sample = sample_li(p, nor, lights, num_lights, geoms, num_geoms, rng);

    if (!sample || sample->pdf == 0.f) {
        return glm::vec3(0.f);
    }

    glm::vec3 bsdf = eval_bsdf(p, nor, -path.ray.dir, sample->wi, m);
    float lambert = glm::max(0.f, glm::dot(sample->wi, nor));

    return sample->radiance * bsdf * lambert / sample->pdf;
}

__global__ void shade_material(int iter, int num_paths, int depth, int num_lights, int num_geoms,
                              const ShadeableIntersection* shadeable_intersections,
                              const MatId* isect_mat_ids, PathSegment* path_segments,
                              const Material* materials, const Light* lights, const Geom* geoms,
                              glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths && path_segments[idx].remaining_bounces > 0) {
        PathSegment path = path_segments[idx];
        MatId mat_id = isect_mat_ids[idx];

        if (mat_id == UINT8_MAX) {
            // hit env map
            path.throughput = glm::vec3(0.0f);
            path.remaining_bounces = 0;
            path_segments[idx] = path;
            return;
        }

        glm::vec3 radiance(0);
        ShadeableIntersection intersection = shadeable_intersections[idx];
        RngEng rng = make_seeded_rng(iter, idx, depth);
        Material material = materials[mat_id];
        glm::vec3 material_color = material.color;

        if (material.type == MatType::Emissive) {
            radiance += path.throughput * material.emission;
            path.remaining_bounces = 0;
        } else {
#if LI_NEE
            glm::vec3 p = get_point_on_ray(path.ray, intersection.t);
            glm::vec3 nor = intersection.surface_normal;
            if (material.type == MatType::Diffuse) {
                // if not delta (so change this when I add microfacet)
                glm::vec3 direct =
                    estimate_direct_lighting(path, p, nor, material, rng, lights, num_lights, geoms, num_geoms);
                radiance += path.throughput * direct;
            }
            scatter_ray(path, p, nor, material, rng);

#elif LI_DIRECT
            glm::vec3 p = get_point_on_ray(path.ray, intersection.t);
            glm::vec3 direct = estimate_direct_lighting(path, p, intersection.surface_normal, material, rng, lights,
                                         num_lights, geoms, num_geoms);
            radiance += direct;
            path.remaining_bounces = 0;
#else
            glm::vec3 intersect = get_point_on_ray(path.ray, intersection.t);
            scatter_ray(path, intersect, intersection.surface_normal, material, rng);
#endif
        }

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
}
