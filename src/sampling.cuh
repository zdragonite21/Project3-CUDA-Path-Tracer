#include <glm/glm.hpp>
#include <thrust/random.h>

__device__ void coordinateSystem(glm::vec3 in_nor, glm::vec3 &out_tan,
                                 glm::vec3 &out_bit);
__device__ glm::vec3
calculateRandomDirectionInCosineHemisphere(glm::vec3 normal,
                                           thrust::default_random_engine &rng);
__device__ glm::vec2 sampleUniformDisk(thrust::default_random_engine &rng);
__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi);
__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi);
__device__ float squareToHemisphereCosinePDF(glm::vec3 s);
__device__ glm::vec3 squareToSphereUniform(glm::vec2 xi);
__device__ float squareToSphereUniformPDF(glm::vec3 s);
