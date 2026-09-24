#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

using MatId = uint8_t;

enum GeomType { SPHERE, CUBE, PLANE };

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
    glm::mat4 invTranspose;
};

struct Geom {
    enum GeomType type;
    int materialId;

    Transform transform;
};

enum class LightType { AREA, POINT, ENVIRONMENT };

struct Light {
    LightType type;

    int geomId;
    glm::vec3 intensity;
};

enum class MatType : uint8_t { DIFFUSE, CONDUCTOR, DIELECTRIC, EMISSIVE };

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
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    glm::vec3 surfaceNormal;
    float t;
};

enum class BxDFFlag : uint8_t {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
};

// overload operators because BxFFlag is strongly typed
__host__ __device__ constexpr BxDFFlag operator|(BxDFFlag a, BxDFFlag b) {
    return static_cast<BxDFFlag>(
        static_cast<uint8_t>(a) | static_cast<uint8_t>(b)
    );
}

__host__ __device__ constexpr BxDFFlag operator&(BxDFFlag a, BxDFFlag b) {
    return static_cast<BxDFFlag>(
        static_cast<uint8_t>(a) & static_cast<uint8_t>(b)
    );
}

__host__ __device__ constexpr BxDFFlag& operator|=(BxDFFlag& a, BxDFFlag b) {
    return a = a | b;
}

struct BSDFSample {
    glm::vec3 wi;
    float pdf;
    glm::vec3 f;
    BxDFFlag type;
};
