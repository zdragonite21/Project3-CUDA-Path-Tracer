# Fractal Path Tracer

![alt text](img/mandelbox_blue_metallic.png)
![alt text](img/mandelbox_light_glass.png)
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
- H to recenter camera at original scene point
- Ctrl + S to writes the scene settings to `/scene-saves`
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

Material sorting is off by default, since it was slower in my [tests](#material-sorting). Time is total kernel time from Nsight Systems per iteration.

### MIS + NEE

![alt text](profiling/graph_mis.png)

|      |                        off                        |                        on                        |
| :--: | :-----------------------------------------------: | :----------------------------------------------: |
| open | ![alt text](profiling/cornell_mis_off/render.png) | ![alt text](profiling/cornell_mis_on/render.png) |

The graphs show that MIS costs about 1 ms more per iteration, but the image converges significantly faster (the right image has visibly less noise).

### compaction

![alt text](profiling/graph_compaction.png)

|        |                             off                              |                             on                              |
| :----: | :----------------------------------------------------------: | :---------------------------------------------------------: |
|  open  |    ![alt text](profiling/cornell_compact_off/render.png)     |    ![alt text](profiling/cornell_compact_on/render.png)     |
| closed | ![alt text](profiling/cornell_closed_compact_off/render.png) | ![alt text](profiling/cornell_closed_compact_on/render.png) |

The open box with compaction is about 30% faster but the closed box is about 5% slower. This is likely because rays in the open box escape and terminate when they hit the HDRI, while rays in the closed box only termiante via russian roulette, so the overhead of performing the compaction (non-stable) outweighs the benefit. 


### material sorting
![alt text](profiling/graph_sort.png)

|                            off                             |                            on                             |
| :--------------------------------------------------------: | :-------------------------------------------------------: |
|     ![alt text](profiling/cornell_sort_off/render.png)     |     ![alt text](profiling/cornell_sort_on/render.png)     |
| ![alt text](profiling/disney_showcase_sort_off/render.png) | ![alt text](profiling/disney_showcase_sort_on/render.png) |

Material sorting is significantly slower for each scene (including the disney showcase, which has 12 materials), likely because the overhead of performing radix sort (on `uint8_t` mat ids) and scattering the pathsegments and intersection data (zipped thrust tuple) outweights the potential benefit from reducing warp divergence, especially since our BSDFs are still relatively cheap to compute.

### cuda streams

I used cuda streams for all the kernels I launch (include thrust) so that sorting and compaction can happen asynchronously, but ordered within the stream.

### intrinsics

I didn't do a formal measurement of this: using cuda intrinsics in the distance estimator hot paths (for trig functions and exponents) helped significantly increase the framerate (about 10x, empirically). The CUDA Best Practices article recommended to not enable project wide intrinsics, but to only uses them in specific places.

### other notes
We update the `num_paths` variable each iteration after compacting to reduce the number of blocks we launch, however this performs a synchronization between the device and the host. I plan to profile and determine if this is a bottleneck in the future.

## more renders

|                                            |                                               |
| :----------------------------------------: | :-------------------------------------------: |
|    ![alt text](img/mandelbox_setup.png)    | ![alt text](img/mandelbox_blue_and_black.png) |
|    ![alt text](img/mandelbox_scifi.png)    |     ![alt text](img/green_mandelbox.png)      |
| ![alt text](img/mandelbulb_unfocused.png)  |    ![alt text](img/cinematic_spheres.png)     |
|    ![alt text](img/gold_mandelbulb.png)    |    ![alt text](img/purple_mandelbulb.png)     |
| ![alt text](img/iridescent_mandelbulb.png) | ![alt text](img/metallic_blue_mandelbulb.png) |
| ![alt text](img/mandelbulb_reflection.png) |     ![alt text](img/glass_mandelbulb.png)     |

See `/saves` and `/renders` for more cool renders.

## timelapse

![alt text](gifs/renders_320_github.gif)

## bloopers

|                                             |                                            |
| :-----------------------------------------: | :----------------------------------------: |
|     ![alt text](img/fake_fish_eye.png)      | ![alt text](img/refraction_distortion.png) |
|    ![alt text](img/black_and_white.png)     |   ![alt text](img/black_and_white2.png)    |
| ![alt text](img/mandelbulb_lobotomized.png) |  ![alt text](img/too_few_march_steps.png)  |

## build instructions

requirements: windows, nvidia gpu, CUDA Toolkit 13.0, Visual Studio 2022 (MSVC), CMake and Ninja.

Run these from a VS 2022 cmd from the repo root dir:
```
cmake -S . -B build/ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/ninja
build\ninja\bin\cis565_path_tracer.exe scenes\cornell.json
```

### feature toggles
Found in `src/config.h`
- SORT_PATHS
- COMPACT_TERMINATED
- LI_MIS
- RUSSIAN_ROULETTE

## future features

- [ ] render more fractals!
- [ ] .exr output for hdr images
- [ ] per intersection shading (procedural materials)
- [ ] further optimizations
    - [ ] CUB
    - [ ] env map mis
    - [ ] relaxed sphere tracing
    - [ ] low-discrepancy sampling (sobol + blue noise)
- [ ] light trees
- [ ] homogenous volumetric rendering
- [ ] photon mapping

## references

- my CIS 4610 path tracer
- pbrt v4
- agx tonemapping: https://github.com/bWFuanVzYWth/AgX/blob/main/agx.glsl
- fractal formulas: https://jbaker.graphics/writings/DEC.html
- mandelbox: https://www.shadertoy.com/view/7tfSzB
- disney bsdf math: https://cseweb.ucsd.edu/~tzli/cse272/wi2026/
- artist friendly metallic fresnel: https://jcgt.org/published/0003/04/03/
