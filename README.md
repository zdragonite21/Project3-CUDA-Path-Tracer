Fractal Path Tracer
================
![alt text](img/mandelbox_light_glass.png)
![alt text](img/mandelbox_blue_metallic.png)
![alt text](img/mandelbulb_purple_metallic.png)
![alt text](img/mandelbox_orange_gui.png)
![alt text](img/mandelbox_metallic.png)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**
- Zachary Leong
  - [LinkedIn](https://linkedin.com/in/zleong), [personal website](https://zacharyleong.com)
- Tested on: Windows 11, Ultra 9 185H @ 2.30GHz 32GB, RTX 4060 Laptop (personal)

## controls

- Esc to exit
- S to save an image in `/saves`
- H to recent camera at original scene point
- Ctrl + S to save the scene settings as a json
- Scroll to move zoom forward
- Right mouse:
  - WASD for movement along local axes
  - Space and Shift for up and down along the y-axis respectively
  - Scroll to adjust camera acceleration

## features

- [anti-aliasing & depth of field](#the-camera) (thin lens approx.)
- [fractal distance estimator ray marching](#fractals)
- [Disney BSDF](#disney-bsdf) (Diffuse, Metal, Clearcoat, Glass, Sheen)
    - GGX
    - anistropy
    - emission
- [imgui + scene loading](#scenes) / scene saving
- [environment lighting](#environment-lighting)
- [agx tonemapping](#color-management)

### optimizations
- [mis + nee](#next-event-estimation-and-multiple-importance-sampling) (and separate shadow ray kernel)
- [russian roulette](#russian-roulette)
- [material sorting + path compaction](#material-sorting-and-compaction)
- [intrinsics](#intrinsics) (hot paths during raymarching)
- [CUDA streams](#cuda-streams) (reducing synchronization with Thrust)

## next event estimation and multiple importance sampling
![alt text](img/cornell_pic.png)

### russian roulette

### environment lighting

## disney bsdf

![alt text](img/disney_showcase.png)
![alt text](img/disney_demo.png)

## the camera

anti aliasing

### depth of field

## fractals

### distance estimators
### raymarching

## scenes

### color management

## performance analysis

### material sorting and compaction

#### cuda streams

### intrinsics

## more renders
![alt text](img/mandelbox_setup.png)
![alt text](img/mandelbox_blue_and_black.png)
![alt text](img/mandelbox_scifi.png)
![alt text](img/green_mandelbox.png)
![alt text](img/mandelbulb_unfocused.png)
![alt text](img/cinematic_spheres.png)
![alt text](img/gold_mandelbulb.png)
![alt text](img/glass_mandelbulb.png)
![alt text](img/iridescent_mandelbulb.png)
![alt text](img/metallic_blue_mandelbulb.png)
![alt text](img/mandelbulb_reflection.png)
![alt text](img/purple_mandelbulb.png)
See `/saves` and `/renders` for more cool renders.
## bloopers
![alt text](img/fake_fish_eye.png)
![alt text](img/refraction_distortion.png)
![alt text](img/black_and_white.png)
![alt text](img/too_few_march_steps.png)
![alt text](img/mandelbulb_lobotomized.png)
## build instructions

## references
- my CIS 4610 path tracer
- pbrt v4
- agx tonemapping: https://github.com/bWFuanVzYWth/AgX/blob/main/agx.glsl
- fractal formulas: https://jbaker.graphics/writings/DEC.html
- mandelbox: https://www.shadertoy.com/view/7tfSzB
- disney bsdf math: https://cseweb.ucsd.edu/~tzli/cse272/wi2026/
- artist friendly metallic fresnel: https://jcgt.org/published/0003/04/03/
