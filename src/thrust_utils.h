#include "sceneStructs.h"

void sort_paths(int num_paths, ShadeableIntersection *isects, MatId *matIds,
                PathSegment *paths);

int compact_missed(int num_paths, MatId *matIds);

int compact_terminated(int num_paths, PathSegment *paths);
