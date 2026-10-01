#include "scene.h"

#include "utilities.h"
#include "math_utils.h"

#include "json.hpp"
#include <glm/gtc/matrix_inverse.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <filesystem>
#include <stb_image.h>
#include <stdexcept>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        load_from_json(filename);
        return;
    } else {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::load_hdri_pixels(const std::string& json_name, const std::string& image_name) {
    const std::filesystem::path scene_path = std::filesystem::absolute(json_name);
    const std::filesystem::path image_path =
        (scene_path.parent_path() / image_name).lexically_normal();
    std::cout << "loading hdri: " << image_path.string() << '\n';

    int source_channels{};

    const std::string filename = image_path.string();
    float* decoded = stbi_loadf(filename.c_str(), &env.width, &env.height, &source_channels, 3);
    if (decoded == nullptr) {
        const char* reason = stbi_failure_reason();

        throw std::runtime_error("failed to load hdir " + filename + "\n" +
                                 (reason ? reason : "unknown error"));
    }

    const size_t pixel_count = static_cast<size_t>(env.width) * static_cast<size_t>(env.height);
    env.pixels.resize(pixel_count);

    for (size_t i = 0; i < pixel_count; ++i) {
        env.pixels[i] = glm::vec3(decoded[3 * i + 0], decoded[3 * i + 1], decoded[3 * i + 2]);
    }

    stbi_image_free(decoded);
}

void Scene::load_from_json(const std::string& json_name) {
    std::ifstream f(json_name);
    json data = json::parse(f);
    const auto& lights_data = data["Lights"];
    for (const auto& item : lights_data.items()) {
        const auto& name = item.key();
        const auto& p = item.value();
        Light new_light{};
        if (p["TYPE"] == "Environment") {
            new_light.geom_id = -1;
            new_light.type = LightType::Environment;
            load_hdri_pixels(json_name, p["PATH"]);
            env.strength = p["STRENGTH"];
            env.light_idx = lights.size();
            lights.push_back(new_light);
        }
    }

    const auto& materials_data = data["Materials"];
    std::unordered_map<std::string, uint32_t> mat_name_to_id;
    for (const auto& item : materials_data.items()) {
        const auto& name = item.key();
        const auto& p = item.value();
        Material new_material{};
        if (p["TYPE"] == "Diffuse") {
            const auto& col = p["RGB"];
            new_material.type = MatType::Diffuse;
            new_material.color = glm::vec3(col[0], col[1], col[2]);
        } else if (p["TYPE"] == "Emitting") {
            const auto& col = p["EMISSION"];
            new_material.type = MatType::Emissive;
            new_material.emission =
                glm::vec3(col[0], col[1], col[2]) * static_cast<float>(p["STRENGTH"]);
        } else if (p["TYPE"] == "Conductor") {
            const auto& eta = p["ETA"];
            const auto& k = p["K"];
            new_material.type = MatType::Conductor;
            new_material.eta = glm::vec3(eta[0], eta[1], eta[2]);
            new_material.k = glm::vec3(k[0], k[1], k[2]);
            new_material.roughness = p["ROUGHNESS"];
        } else if (p["TYPE"] == "Dielectric") {
            new_material.type = MatType::Dielectric;
            new_material.roughness = p["ROUGHNESS"];
            new_material.ior = p["IOR"];
        }
        mat_name_to_id[name] = materials.size();
        materials.push_back(new_material);
    }
    const auto& objects_data = data["Objects"];
    for (const auto& p : objects_data) {
        const auto& type = p["TYPE"];
        Geom new_geom{};
        if (type == "cube") {
            new_geom.type = Cube;
        } else if (type == "plane") {
            new_geom.type = Plane;
        } else if (type == "sdf") {
            new_geom.type = Sdf;
        } else {
            new_geom.type = Sphere;
        }
        new_geom.material_id = mat_name_to_id[p["MATERIAL"]];
        if (materials[new_geom.material_id].type == MatType::Emissive) {
            Light new_light{};
            new_light.geom_id = geoms.size();
            new_light.type = LightType::Area;
            new_light.emission = materials[new_geom.material_id].emission;
            new_geom.light_idx = lights.size();
            lights.push_back(new_light);
        }
        Transform new_trans;
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        new_trans.translation = glm::vec3(trans[0], trans[1], trans[2]);
        new_trans.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        new_trans.scale = glm::vec3(scale[0], scale[1], scale[2]);
        new_trans.matrix = utility_core::build_transformation_matrix(
            new_trans.translation, new_trans.rotation, new_trans.scale);
        new_trans.inverse = glm::inverse(new_trans.matrix);
        new_trans.inv_transpose = glm::inverseTranspose(new_trans.matrix);
        new_geom.transform = new_trans;

        geoms.push_back(new_geom);
    }
    const auto& camera_data = data["Camera"];
    CameraData& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = camera_data["RES"][0];
    camera.resolution.y = camera_data["RES"][1];
    float fovy = camera_data["FOVY"];
    camera.lens_radius = camera_data["LENSRADIUS"];
    camera.focal_distance = camera_data["FOCALDISTANCE"];
    state.iterations = camera_data["ITERATIONS"];
    state.trace_depth = camera_data["DEPTH"];
    state.image_name = camera_data["FILE"];
    const auto& pos = camera_data["EYE"];
    const auto& lookat = camera_data["LOOKAT"];
    const auto& up = camera_data["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.look_at = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.pixel_length = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                    2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.look_at - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3(0));
}
