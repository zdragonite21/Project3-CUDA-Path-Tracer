#include <glm/glm.hpp>
#include <thrust/random.h>

__device__ glm::vec3
calculateRandomDirectionInCosineHemisphere(thrust::default_random_engine &rng);
__device__ glm::vec2 sampleUniformDisk(thrust::default_random_engine &rng);
__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi);
__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi);
__device__ float squareToHemisphereCosinePDF(glm::vec3 s);
__device__ glm::vec3 squareToSphereUniform(glm::vec2 xi);
__device__ float squareToSphereUniformPDF(glm::vec3 s);
