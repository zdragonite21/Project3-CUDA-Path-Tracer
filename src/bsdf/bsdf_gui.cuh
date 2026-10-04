#pragma once
#include "bsdf_structs.cuh"
#include "imgui.h"

static bool draw_bsdf(Lambertian& m) {
    return ImGui::ColorEdit3("color", &m.color.x);
}
static bool draw_bsdf(Conductor& m) {
    bool r = false;
    r |= ImGui::DragFloat3("eta", &m.eta.x, 0.01f, 0.f, 10.f);
    r |= ImGui::DragFloat3("k", &m.k.x, 0.01f, 0.f, 10.f);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("anisotropy", &m.anisotropy, -1.f, 1.f);
    return r;
}
static bool draw_bsdf(Dielectric& m) {
    bool r = false;
    r |= ImGui::SliderFloat("ior", &m.ior, 1.f, 3.f);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    return r;
}
static bool draw_bsdf(DisneyDiffuse& m) {
    bool r = false;
    r |= ImGui::ColorEdit3("color", &m.color.x);
    r |= ImGui::SliderFloat("roughness", &m.roughness, 0.f, 1.f);
    r |= ImGui::SliderFloat("subsurface", &m.subsurface, 0.f, 1.f);
    return r;
}
