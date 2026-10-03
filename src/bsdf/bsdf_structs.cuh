// #pragma once

// #include <cuda_runtime.h>
// #include <cuda/std/variant>

// struct Lambertian    { glm::vec3 color; };
// struct Conductor     { glm::vec3 eta, k; float roughness; };
// struct Dielectric    { float ior, roughness; };
// struct DisneyDiffuse { glm::vec3 base_color; float roughness, subsurface; };
// struct DisneyMetal {
//     glm::vec3 base_color;
//     float roughness, anisotropic;
// };

// using BsdfVariant = cuda::std::variant<Lambertian, Conductor, Dielectric,
//                                        DisneyDiffuse, DisneyMetal /*, ...*/>;
