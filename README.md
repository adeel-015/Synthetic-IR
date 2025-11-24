# Synthetic IR Dataset Generation Assignment

## Overview

This project generates a synthetic Infrared (IR) dataset using a physics-based procedural approach. The goal is to simulate thermal behavior and sensor characteristics to demonstrate how IR imaging differs from visible light imaging and to provide a realistic dataset for algorithm development (detection, tracking, GAN training).

## Methodology

### 1. Thermodynamics & Temperature Mapping

Instead of generating random pixel values, the simulation first creates a floating-point "Temperature Grid" representing scene temperature in Celsius. This grid is then converted into pixel intensities by the sensor model.

- **Background:** Modeled as a consistent ambient temperature of **20°C** with small random fluctuations to simulate air turbulence.
- **Objects:** Modeled as heat sources that may range from biological temperatures (~36.5°C) to mechanical sources (~50°C).
- **Mapping:** Temperature values in the range **15°C to 55°C** are linearly mapped to the 8-bit intensity range 0–255 using the radiance-to-intensity mapping documented below.

**Emissivity & Material Behavior**

- For simplicity all objects in this dataset are treated as perfect emitters (emissivity ε = 1.0), i.e., black-body approximation. This makes the simulation simpler and reproducible.
- Real-world materials have emissivity < 1.0 (e.g., metals ε≈0.1–0.3, human skin ε≈0.98). In practical deployments a per-material emissivity map should be used to correct raw radiance values.

### 2. IR Sensor Effects Modeled

To make the data realistic for algorithm development, the following sensor characteristics are simulated:

- **Thermal Diffusion (Blur):** A 5×5 Gaussian blur is applied to the temperature grid to simulate atmospheric scattering and the optical point-spread function of the lens.
- **Sensor Noise:** Additive Gaussian noise (σ = 2.0) is applied after normalization and scaling to simulate microbolometer read noise and thermal fluctuations.
- **Bit-Depth & Quantization:** Final float intensities are clipped and quantized into **8-bit integers** (0–255), modeling the ADC process.

**Radiance-to-Intensity Mapping**

- The script uses a linear mapping from temperature to pixel intensity for the sensor dynamic range. The formula used is:

```
pixel_intensity = (T - T_min) / (T_max - T_min) × 255
```

- The simulation maps temperatures between **15°C (T_min)** and **55°C (T_max)** to the 0–255 intensity range. This is a linear calibration approximation; real IR radiance follows Stefan–Boltzmann's law (radiance ∝ T⁴), but a linear mapping is a reasonable simplification over small temperature spans.

### 3. Temporal Variation

- **Motion:** Objects move linearly across the frame using velocity vectors. Boundary handling uses toroidal wrapping (objects re-enter on the opposite edge).
- **Cooling:** Objects can have a `cooling_rate` (°C per frame) to simulate dissipating heat.
- **Temperature Pulsing:** Objects may optionally include `pulse_amplitude` and `pulse_frequency` values. The simulation adds a sinusoidal temperature variation per-frame (used to simulate breathing, engine cycles, or thermal regulation).

## Sensor Characteristics (Detailed)

| Property                |                                                           Value |
| ----------------------- | --------------------------------------------------------------: |
| Resolution              |                                                  256×256 pixels |
| Bit Depth               |                                                   8-bit (0–255) |
| Noise Model             |                                      Additive Gaussian, σ = 2.0 |
| Blur Kernel             |                              5×5 Gaussian (optical/atmospheric) |
| Dynamic Range           |                                           15°C–55°C (40°C span) |
| Frames                  |                                             30 frames generated |
| Atmospheric Attenuation | Distance-based exponential attenuation applied along the y-axis |

### Sensor Effects Implementation Details

- **Thermal Diffusion (Blur):** A 5×5 Gaussian convolution is applied to the temperature grid prior to normalization to simulate diffusion and optical PSF.
- **Sensor Noise:** After scaling to the 8-bit range, additive Gaussian noise (σ = 2.0) is applied to each pixel then the result is clipped and quantized to uint8.
- **Bit-Depth Quantization:** Float32 intensities are clipped to 0–255 and converted to `uint8` to simulate ADC quantization.
- **Atmospheric Attenuation:** A distance-based exponential attenuation is applied to simulate absorption/scattering with increasing distance. The per-pixel attenuation is computed from the pixel row (y) and the thermal map is multiplied element-wise by an exponential decay.

## How to Run

1. Install dependencies: `pip install numpy opencv-python`
2. Run the script: `python generate_ir_data.py`
3. Output frames will appear in the `ir_dataset/` directory.

The script generates 30 frames containing diverse object types (Gaussian blobs, rectangles, silhouettes) exhibiting motion, temperature pulsing, cooling, and realistic sensor effects (blur, noise, distance attenuation). Use the generated frames for prototyping and algorithm validation.

## Limitations

- Background clutter is still simplistic and mostly uniform.
- Non-uniform detector response (fixed pattern noise) is not modeled.
