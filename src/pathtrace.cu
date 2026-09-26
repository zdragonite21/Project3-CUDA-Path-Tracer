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
#include "sceneStructs.h"
#include "thrust_utils.h"
#include "utilities.h"

#define ERRORCHECK 0

#define SORT_PATHS 1
#define COMPACT_TERMINATED 1

#define RUSSIAN_ROULETTE 1

#define LI_NEE 1
#define LI_DIRECT 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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

__host__ __device__ RngEng makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return RngEng(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
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
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Light* dev_lights = NULL;
static MatId* dev_isect_matIds = NULL;

cudaStream_t pt_stream;

void InitDataContainer(GuiDataContainer* imGuiData) {
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
    const Camera& cam = scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_isect_matIds, pixelcount * sizeof(MatId));

    cudaStreamCreate(&pt_stream);

    checkCUDAError("pathtraceInit");
}

void pathtraceReset(Scene* scene) {
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
               cudaMemcpyHostToDevice);

    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light),
               cudaMemcpyHostToDevice);

    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material),
               cudaMemcpyHostToDevice);

    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_isect_matIds, 0, pixelcount * sizeof(MatId));

    checkCUDAError("pathtraceReset");
}

void pathtraceFree() {
    cudaStreamSynchronize(pt_stream);

    cudaFree(dev_image); // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_isect_matIds);
    cudaFree(dev_lights);

    cudaStreamDestroy(pt_stream);

    checkCUDAError("pathtraceFree");
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
                                      PathSegment* pathSegments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        RngEng rng = makeSeededRandomEngine(iter, index, 0);
        UnifDist<float> u01(0, 1);

        glm::vec2 offset = glm::vec2(u01(rng), u01(rng));
        glm::vec2 sub_pixel_sample = glm::vec2(x, y) + offset;

        Ray& ray = segment.ray;
        ray.org = cam.position;
        ray.dir = glm::normalize(
            cam.view -
            cam.right * cam.pixelLength.x * (sub_pixel_sample.x - (float)cam.resolution.x * 0.5f) -
            cam.up * cam.pixelLength.y * (sub_pixel_sample.y - (float)cam.resolution.y * 0.5f));

        if (cam.lensRadius > 0.0) {
            float t = cam.focalDistance / glm::dot(ray.dir, cam.view);
            glm::vec3 pFocus = ray.org + ray.dir * t;
            glm::vec2 pLens = cam.lensRadius * sampleUniformDisk(rng);
            ray.org += cam.right * pLens.x + cam.up * pLens.y;
            ray.dir = glm::normalize(pFocus - ray.org);
        }

        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(int depth, int num_paths, PathSegment* pathSegments,
                                     Geom* geoms, int geoms_size,
                                     ShadeableIntersection* intersections, MatId* isect_matIds) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        Ray r = pathSegments[path_index].ray;

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++) {
            Geom& geom = geoms[i];

            if (geom.type == CUBE) {
                t = boxIntersectionTest(geom, r, &tmp_intersect, &tmp_normal, &outside);
            } else if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, r, &tmp_intersect, &tmp_normal, &outside);
            } else if (geom.type == PLANE) {
                t = planeIntersectionTest(geom, r, &tmp_intersect, &tmp_normal, &outside);
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
            isect_matIds[path_index] = UINT8_MAX;
        } else {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surfaceNormal = normal;
            isect_matIds[path_index] = geoms[hit_geom_index].materialId;
        }
    }
}

__global__ void shadeMaterial(int iter, int num_paths, int depth, int lights_size, int geoms_size,
                              ShadeableIntersection* shadeableIntersections, MatId* isect_matIds,
                              PathSegment* pathSegments, Material* materials, Light* lights,
                              Geom* geoms, glm::vec3* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths && pathSegments[idx].remainingBounces > 0) {
        PathSegment path = pathSegments[idx];
        MatId matId = isect_matIds[idx];

        if (matId == UINT8_MAX) {
            // hit env map
            path.throughput = glm::vec3(0.0f);
            path.remainingBounces = 0;
            pathSegments[idx] = path;
            return;
        }

        glm::vec3 radiance(0);
        ShadeableIntersection intersection = shadeableIntersections[idx];
        RngEng rng = makeSeededRandomEngine(iter, idx, depth);
        Material material = materials[matId];
        glm::vec3 materialColor = material.color;

        if (material.type == MatType::EMISSIVE) {
            radiance += path.throughput * material.emission;
            path.remainingBounces = 0;
        } else {
#if LI_NEE
            glm::vec3 p = getPointOnRay(path.ray, intersection.t);
            glm::vec3 nor = intersection.surfaceNormal;
            if (material.type == MatType::DIFFUSE) {
                // if not delta (so change this when I add microfacet)
                glm::vec3 direct =
                    directRay(path, p, nor, material, rng, lights, lights_size, geoms, geoms_size);
                radiance += path.throughput * direct;
            }
            scatterRay(path, p, nor, material, rng);

#elif LI_DIRECT
            glm::vec3 p = getPointOnRay(path.ray, intersection.t);
            glm::vec3 direct = directRay(path, p, intersection.surfaceNormal, material, rng, lights,
                                         lights_size, geoms, geoms_size);
            radiance += direct;
            path.remainingBounces = 0;
#else
            glm::vec3 intersect = getPointOnRay(path.ray, intersection.t);
            scatterRay(path, intersect, intersection.surfaceNormal, material, rng);
#endif
        }

#if RUSSIAN_ROULETTE
        if (depth > 3 && path.remainingBounces > 0) {
            UnifDist<float> u01(0, 1);

            float surviveP = max(path.throughput.x, max(path.throughput.y, path.throughput.z));
            surviveP = glm::clamp(surviveP, 0.05f, 0.95f);
            if (u01(rng) > surviveP) {
                path.throughput = glm::vec3(0);
                path.remainingBounces = 0;
            } else {
                path.throughput /= surviveP;
            }
        }
#endif
        pathSegments[idx] = path;
        // final gather
        image[path.pixelIndex] += radiance;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 * Lo(p, wo) = Le(p, wo) + 1/n * sum(bsdf(p, wo, wi) * Li(p, wi) * absdot(wi,
 * nor) / pdf(wi))
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d((cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
                               (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d, 0, pt_stream>>>(cam, iter, traceDepth,
                                                                          dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    bool iterationComplete = false;
    while (!iterationComplete && depth < traceDepth) {
        // tracing
        dim3 numblocksPathSegmentTracing = utilityCore::divup(num_paths, blockSize1d);
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d, 0, pt_stream>>>(
            depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections,
            dev_isect_matIds);
        checkCUDAError("trace one bounce");
        depth++;

#if SORT_PATHS
        sort_paths(num_paths, dev_intersections, dev_isect_matIds, dev_paths, pt_stream);
#endif
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d, 0, pt_stream>>>(
            iter, num_paths, depth, hst_scene->lights.size(), hst_scene->geoms.size(),
            dev_intersections, dev_isect_matIds, dev_paths, dev_materials, dev_lights, dev_geoms,
            dev_image);
        checkCUDAError("shader material");

#if COMPACT_TERMINATED
#if SORT_PATHS
        // use binary search to filer down paths since we already did the sort
        num_paths = filter_missed(num_paths, dev_isect_matIds, pt_stream);
#endif
        // compact the remaning terminated paths
        num_paths = compact_terminated(num_paths, dev_paths, pt_stream);
#endif
        if (num_paths < 1) {
            iterationComplete = true;
        }

        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d, 0, pt_stream>>>(pbo, cam.resolution, iter,
                                                                   dev_image);
    checkCUDAError("pathtrace");
}

void copyImageToHost() {
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);
}
