#pragma once

#include "scene_structs.h"
#include <vector>

struct EnvironmentMap {
    int width = 0;
    int height = 0;
    float strength = 1.f;
    int light_idx = -1;
    std::vector<glm::vec3> pixels;
};

class Scene
{
  private:
    void load_hdri_pixels(const std::string& json_name, const std::string& hdri_path);
    void load_from_json(const std::string& json_name);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;
    RenderState state;
    EnvironmentMap env;
};
