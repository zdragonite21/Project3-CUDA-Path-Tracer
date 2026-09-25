#include <glm/glm.hpp>
#include "thrust_utils.h"

__device__ glm::vec3
calculateRandomDirectionInCosineHemisphere(RngEng &rng);
__device__ glm::vec2 sampleUniformDisk(RngEng &rng);
__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi);
__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi);
__device__ float squareToHemisphereCosinePDF(glm::vec3 s);
__device__ glm::vec3 squareToSphereUniform(glm::vec2 xi);
__device__ float squareToSphereUniformPDF(glm::vec3 s);
