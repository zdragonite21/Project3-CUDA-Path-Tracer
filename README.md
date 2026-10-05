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

### features
- depth of field (thin lens approx.)
- material sorting + path compaction
- anti-aliasing
- russian roulette
- mis + nee (and separate shadow ray kernel)
- fractal distance estimator ray marching
- Disney BSDF (Diffuse, Metal, Clearcoat, Glass, Sheen)
    - GGX
    - anistropy
    - emission
- imgui + scene loading / scene saving
- environment lighting
- agx tonemapping

### references

- My CIS 4610 path tracer
- https://github.com/bWFuanVzYWth/AgX/blob/main/agx.glsl
- https://jbaker.graphics/writings/DEC.html
- https://cseweb.ucsd.edu/~tzli/cse272/wi2026/
- https://jcgt.org/published/0003/04/03/ (Artist Friendly Metallic Fresnel)
