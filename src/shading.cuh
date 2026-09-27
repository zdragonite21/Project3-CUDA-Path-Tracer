#pragma once

#include "scene_structs.h"

__global__ void shade_material(int iter, int num_paths, int depth, int num_lights, int num_geoms,
                              const ShadeableIntersection* shadeable_intersections,
                              const MatId* isect_mat_ids, PathSegment* path_segments,
                              const Material* materials, const Light* lights, const Geom* geoms,
                              glm::vec3* image);
