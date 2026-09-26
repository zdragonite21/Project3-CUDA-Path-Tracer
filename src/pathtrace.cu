#include "pathtrace.h"

#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "interactions.h"
#include "intersections.h"
#include "sampling.cuh"
#include "scene.h"
#include "scene_structs.h"
#include "thrust_utils.h"
#include "utilities.h"

#define ERRORCHECK 0

#define SORT_PATHS 1
#define COMPACT_TERMINATED 1

#define RUSSIAN_ROULETTE 1

#define LI_NEE 1
#define LI_DIRECT 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define check_cuda_error(msg) check_cuda_error_fn(msg, FILENAME, __LINE__)
void check_cuda_error_fn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__ RngEng make_seeded_rng(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return RngEng(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void send_image_to_pbo(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
    if (iter == 0) {
        return;
    }

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* gui_data = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Light* dev_lights = NULL;
static MatId* dev_isect_mat_ids = NULL;

cudaStream_t pt_stream;

void init_data_container(GuiDataContainer* imgui_data) {
    gui_data = imgui_data;
}

void pathtrace_init(Scene* scene) {
    const Camera& cam = scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, num_pixels * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, num_pixels * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));

    cudaMalloc(&dev_intersections, num_pixels * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_isect_mat_ids, num_pixels * sizeof(MatId));

    cudaStreamCreate(&pt_stream);

    check_cuda_error("pathtrace_init");
}

void pathtrace_reset(Scene* scene) {
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;

    cudaMemset(dev_image, 0, num_pixels * sizeof(glm::vec3));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
               cudaMemcpyHostToDevice);

    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light),
               cudaMemcpyHostToDevice);

    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
               cudaMemcpyHostToDevice);

    cudaMemset(dev_intersections, 0, num_pixels * sizeof(ShadeableIntersection));
    cudaMemset(dev_isect_mat_ids, 0, num_pixels * sizeof(MatId));

    check_cuda_error("pathtrace_reset");
}

void pathtrace_free() {
    cudaStreamSynchronize(pt_stream);

    cudaFree(dev_image); // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_isect_mat_ids);
    cudaFree(dev_lights);

    cudaStreamDestroy(pt_stream);

    check_cuda_error("pathtrace_free");
}

__global__ void gen_ray_from_cam(Camera cam, int iter, int trace_depth,
                                      PathSegment* path_segments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = path_segments[index];

        RngEng rng = make_seeded_rng(iter, index, 0);
        UnifDist<float> u01(0, 1);

        glm::vec2 offset = glm::vec2(u01(rng), u01(rng));
        glm::vec2 sub_pixel_sample = glm::vec2(x, y) + offset;

        Ray& ray = segment.ray;
        ray.org = cam.position;
        ray.dir = glm::normalize(
            cam.view -
            cam.right * cam.pixel_length.x * (sub_pixel_sample.x - (float)cam.resolution.x * 0.5f) -
            cam.up * cam.pixel_length.y * (sub_pixel_sample.y - (float)cam.resolution.y * 0.5f));

        if (cam.lens_radius > 0.0) {
            float t = cam.focal_distance / glm::dot(ray.dir, cam.view);
            glm::vec3 p_focus = ray.org + ray.dir * t;
            glm::vec2 p_lens = cam.lens_radius * sample_uniform_disk(rng);
            ray.org += cam.right * p_lens.x + cam.up * p_lens.y;
            ray.dir = glm::normalize(p_focus - ray.org);
        }

        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixel_index = index;
        segment.remaining_bounces = trace_depth;
    }
}

__global__ void compute_intersections(int num_paths, const PathSegment* path_segments,
                                     const Geom* geoms, int num_geoms,
                                     ShadeableIntersection* intersections, MatId* isect_mat_ids) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        Ray r = path_segments[path_index].ray;

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms
        for (int i = 0; i < num_geoms; i++) {
            const Geom& geom = geoms[i];

            if (geom.type == Cube) {
                t = box_intersection_test(geom, r, &tmp_intersect, &tmp_normal, &outside);
            } else if (geom.type == Sphere) {
                t = sphere_intersection_test(geom, r, &tmp_intersect, &tmp_normal, &outside);
            } else if (geom.type == Plane) {
                t = plane_intersection_test(geom, r, &tmp_intersect, &tmp_normal, &outside);
                // only intersect with one side
                if (outside) {
                    continue;
                }
            }

            // Compute the minimum t from the intersection tests to determine
            // what scene geometry object was hit first.
            if (t > 0.0f && t_min > t) {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1) {
            // hit env map
            intersections[path_index].t = -1.0f;
            isect_mat_ids[path_index] = UINT8_MAX;
        } else {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surface_normal = normal;
            isect_mat_ids[path_index] = geoms[hit_geom_index].material_id;
        }
    }
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 * Lo(p, wo) = Le(p, wo) + 1/n * sum(bsdf(p, wo, wi) * Li(p, wi) * absdot(wi,
 * nor) / pdf(wi))
 */
void pathtrace(uchar4* pbo, int iter) {
    const int trace_depth = hst_scene->state.trace_depth;
    const Camera& cam = hst_scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 block_size_2d(8, 8);
    const dim3 blocks_per_grid_2d((cam.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
                               (cam.resolution.y + block_size_2d.y - 1) / block_size_2d.y);

    // 1D block for path tracing
    const int block_size_1d = 128;

    gen_ray_from_cam<<<blocks_per_grid_2d, block_size_2d, 0, pt_stream>>>(cam, iter, trace_depth,
                                                                          dev_paths);
    check_cuda_error("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + num_pixels;
    int num_paths = dev_path_end - dev_paths;

    bool iteration_complete = false;
    while (!iteration_complete && depth < trace_depth) {
        // tracing
        dim3 num_blocks_path_segment_tracing = utility_core::divup(num_paths, block_size_1d);
        compute_intersections<<<num_blocks_path_segment_tracing, block_size_1d, 0, pt_stream>>>(
            num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections,
            dev_isect_mat_ids);
        check_cuda_error("trace one bounce");
        depth++;

#if SORT_PATHS
        sort_paths(num_paths, dev_intersections, dev_isect_mat_ids, dev_paths, pt_stream);
#endif
        shade_material<<<num_blocks_path_segment_tracing, block_size_1d, 0, pt_stream>>>(
            iter, num_paths, depth, hst_scene->lights.size(), hst_scene->geoms.size(),
            dev_intersections, dev_isect_mat_ids, dev_paths, dev_materials, dev_lights, dev_geoms,
            dev_image);
        check_cuda_error("shader material");

#if COMPACT_TERMINATED
#if SORT_PATHS
        // use binary search to filer down paths since we already did the sort
        num_paths = filter_missed(num_paths, dev_isect_mat_ids, pt_stream);
#endif
        // compact the remaning terminated paths
        num_paths = compact_terminated(num_paths, dev_paths, pt_stream);
#endif
        if (num_paths < 1) {
            iteration_complete = true;
        }

        if (gui_data != NULL) {
            gui_data->traced_depth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    send_image_to_pbo<<<blocks_per_grid_2d, block_size_2d, 0, pt_stream>>>(pbo, cam.resolution, iter,
                                                                   dev_image);
    check_cuda_error("pathtrace");
}

void copy_image_to_host() {
    const Camera& cam = hst_scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), dev_image, num_pixels * sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);
}
