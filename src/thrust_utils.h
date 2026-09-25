#pragma once

#include "sceneStructs.h"
#include <thrust/random.h>

using RngEng = thrust::default_random_engine;
template <typename T>
using UnifDist = thrust::uniform_real_distribution<T>;

void sort_paths(int num_paths, ShadeableIntersection *isects, MatId *matIds,
                PathSegment *paths, cudaStream_t stream);

int filter_missed(int num_paths, MatId *matIds, cudaStream_t stream);

int compact_terminated(int num_paths, PathSegment* paths, cudaStream_t stream);
