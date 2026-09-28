#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

using MatId = uint8_t;

enum GeomType { Sphere, Cube, Plane };

struct Ray {
    glm::vec3 org;
    glm::vec3 dir;
};

struct Transform {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 matrix;
    glm::mat4 inverse;
    glm::mat4 inv_transpose;
};

struct Geom {
    enum GeomType type;
    int material_id;
    int light_idx;

    Transform transform;
};

enum class LightType { Area, Environment };

// for area lights, both light and material get the same emission (for convenience)
// light -> geom -> material
struct Light {
    glm::vec3 emission;
    int geom_id;
    
    float env_strength;

    LightType type;
};

enum class MatType : uint8_t { Diffuse, Conductor, Dielectric, Emissive };

struct Material {
    glm::vec3 color;
    glm::vec3 eta;
    glm::vec3 k;

    float roughness;
    float ior;

    glm::vec3 emission;

    MatType type;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 look_at;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixel_length;
    float lens_radius;
    float focal_distance;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int trace_depth;
    std::vector<glm::vec3> image;
    std::string image_name;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    int pixel_index;
    int remaining_bounces;
    float prev_bsdf_pdf;
    bool prev_was_delta;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    glm::vec3 surface_normal;
    float t;
    int light_idx;
};

enum class BxdfFlag : uint8_t {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
};

// overload operators because BxdfFlag is strongly typed
__host__ __device__ constexpr BxdfFlag operator|(BxdfFlag a, BxdfFlag b) {
    return static_cast<BxdfFlag>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

__host__ __device__ constexpr BxdfFlag operator&(BxdfFlag a, BxdfFlag b) {
    return static_cast<BxdfFlag>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

__host__ __device__ constexpr BxdfFlag& operator|=(BxdfFlag& a, BxdfFlag b) {
    return a = a | b;
}

struct BsdfSample {
    glm::vec3 wi;
    float pdf;
    glm::vec3 f;
    BxdfFlag type;
};

struct LightSample {
    glm::vec3 wi;
    float dist;
    float pdf;
    int light_idx;
};

struct ShadowRay {
    Ray ray;
    int pixel_index;
    glm::vec3 contribution;
    float t_max;
};
