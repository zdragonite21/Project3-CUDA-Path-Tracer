#include "scene.h"

#include "utilities.h"

#include "json.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json") {
        loadFromJSON(filename);
        return;
    } else {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string &jsonName) {
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto &lightsData = data["Lights"];
    for (const auto &item : lightsData.items()) {
        const auto &name = item.key();
        const auto &p = item.value();
        Light newLight{};
        if (p["TYPE"] == "Environment") {
            newLight.geom_id = -1;
            newLight.type = LightType::Environment;
            newLight.env_strength = p["STRENGTH"];
        }
        // TODO: integrate environment lighting
        // lights.push_back(newLight);
    }

    const auto &materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto &item : materialsData.items()) {
        const auto &name = item.key();
        const auto &p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse") {
            const auto &col = p["RGB"];
            newMaterial.type = MatType::Diffuse;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        } else if (p["TYPE"] == "Emitting") {
            const auto &col = p["EMISSION"];
            newMaterial.type = MatType::Emissive;
            newMaterial.emission = glm::vec3(col[0], col[1], col[2]) *
                                   static_cast<float>(p["STRENGTH"]);
        } else if (p["TYPE"] == "Conductor") {
            const auto &eta = p["ETA"];
            const auto &k = p["K"];
            newMaterial.type = MatType::Conductor;
            newMaterial.eta = glm::vec3(eta[0], eta[1], eta[2]);
            newMaterial.k = glm::vec3(k[0], k[1], k[2]);
            newMaterial.roughness = p["ROUGHNESS"];
        } else if (p["TYPE"] == "Dielectric") {
            newMaterial.type = MatType::Dielectric;
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.ior = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.push_back(newMaterial);
    }
    const auto &objectsData = data["Objects"];
    for (const auto &p : objectsData) {
        const auto &type = p["TYPE"];
        Geom newGeom;
        if (type == "cube") {
            newGeom.type = Cube;
        } else if (type == "plane") {
            newGeom.type = Plane;
        } else {
            newGeom.type = Sphere;
        }
        newGeom.material_id = MatNameToID[p["MATERIAL"]];
        if (materials[newGeom.material_id].type == MatType::Emissive) {
            Light newLight{};
            newLight.geom_id = geoms.size();
            newLight.type = LightType::Area;
            newLight.emission = materials[newGeom.material_id].emission;
            lights.push_back(newLight);
        }
        Transform newTrans;
        const auto &trans = p["TRANS"];
        const auto &rotat = p["ROTAT"];
        const auto &scale = p["SCALE"];
        newTrans.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newTrans.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newTrans.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newTrans.matrix = utility_core::build_transformation_matrix(
            newTrans.translation, newTrans.rotation, newTrans.scale);
        newTrans.inverse = glm::inverse(newTrans.matrix);
        newTrans.inv_transpose = glm::inverseTranspose(newTrans.matrix);
        newGeom.transform = newTrans;

        geoms.push_back(newGeom);
    }
    const auto &cameraData = data["Camera"];
    Camera &camera = state.camera;
    RenderState &state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    camera.lens_radius = cameraData["LENSRADIUS"];
    camera.focal_distance = cameraData["FOCALDISTANCE"];
    state.iterations = cameraData["ITERATIONS"];
    state.trace_depth = cameraData["DEPTH"];
    state.image_name = cameraData["FILE"];
    const auto &pos = cameraData["EYE"];
    const auto &lookat = cameraData["LOOKAT"];
    const auto &up = cameraData["UP"];
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
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
