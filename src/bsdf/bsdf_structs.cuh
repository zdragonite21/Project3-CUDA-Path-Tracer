#pragma once

#include <cuda/std/variant>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct Lambertian {
    glm::vec3 color;
    __device__ bool is_delta() const {
        return false;
    }
};
struct Conductor {
    glm::vec3 eta;
    glm::vec3 k;
    float roughness;
    float anisotropy;
    __device__ bool is_delta() const {
        return roughness == 0.f;
    }
};
struct Dielectric {
    float ior;
    float roughness;
    __device__ bool is_delta() const {
        return roughness == 0.f;
    }
};

struct DisneyDiffuse {
    glm::vec3 color;
    float roughness;
    float subsurface;
    __device__ bool is_delta() const {
        return false;
    }
};

// struct DisneyMetal {
//     glm::vec3 base_color;
//     float roughness;
//     float anisotropic;
//     __device__ bool is_delta() const {
//         return false;
//     }
// };

using BsdfVariant = cuda::std::variant<Lambertian, Conductor, Dielectric, DisneyDiffuse>;
