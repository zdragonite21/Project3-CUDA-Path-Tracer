#pragma once

#include "scene_structs.h"
#include <vector>

class Scene
{
private:
    void load_from_json(const std::string& json_name);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;
    RenderState state;
};
