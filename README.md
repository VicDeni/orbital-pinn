# Vision-Assisted Physics-Informed Neural Network for Satellite Orbit Estimation

A three-part project exploring how physics-informed machine learning, computer vision, and adaptive sensor fusion can work together to predict and correct satellite orbital trajectories.

## Overview

Traditional orbital simulations solve Newton's equations of motion through numerical integration. This project investigates an alternative: a **Physics-Informed Neural Network (PINN)** that learns orbital trajectories by embedding the governing differential equations directly into its training process, rather than learning purely from labeled data.

The project then extends this with a **vision-assisted correction layer** simulating how real orbit-tracking systems fuse physics-based predictions with observational data, and finally replaces a naive fixed-ratio correction with an **Extended Kalman Filter** — the principled, adaptive technique real spacecraft tracking systems actually use.

## Project Structure

### Phase 1 — Physics-Informed Neural Network
`notebooks/01_orbital_mechanics_baseline.ipynb`

- Simulates a two-body orbital system using classical numerical methods (`scipy.integrate.solve_ivp`) to generate a trusted ground-truth trajectory.
- Builds a PINN (PyTorch) that predicts satellite position from time alone.
- Trains the network using a physics-informed loss function — via automatic differentiation, the network's own predicted acceleration is checked against Newton's law of gravitation, so it learns to obey the governing equation rather than just memorizing data points.
- Resolves a long-horizon training instability using curriculum learning (progressively extending the training time horizon) and explicit energy/angular-momentum conservation constraints.

### Phase 2 — Vision-Assisted Orbit Correction
`notebooks/02_vision_assisted_correction.ipynb`

- Introduces a realistic disturbance (atmospheric drag) so the true orbit no longer matches the PINN's idealized physics.
- Simulates synthetic "telescope images" of the satellite's true position.
- Builds a small computer vision model (CNN) to detect the satellite's position directly from these images, achieving sub-pixel localization accuracy.
- Fuses these vision-based observations into the PINN's prediction using a fixed blending ratio, reducing average position error by ~70% over the physics-only baseline.

### Phase 3 — Adaptive Fusion with an Extended Kalman Filter
`notebooks/03_kalman_filter_fusion.ipynb`

- Replaces Phase 2's fixed blending ratio with an Extended Kalman Filter (EKF), which adaptively weighs trust between the physics prediction and vision observation based on real-time uncertainty.
- Uses PyTorch's automatic differentiation to compute the filter's Jacobian directly from the dynamics model, rather than deriving it by hand.
- Achieves a ~99% reduction in average position error compared to the physics-only baseline, and a ~97% reduction compared to the simple fixed-ratio blend from Phase 2.

## Key Concepts Demonstrated

- Physics-informed machine learning (PINNs) and automatic differentiation
- Classical numerical ODE solving for orbital dynamics
- Custom physics-based loss functions in PyTorch, including conservation-law constraints
- Curriculum learning for long-horizon neural ODE solutions
- Computer vision for object localization (CNN-based detection)
- Sensor fusion via Extended Kalman Filtering
- Interactive and comparative scientific visualization

## Setup

```bash
git clone https://github.com/VicDeni/orbital-pinn.git
cd orbital-pinn
python3 -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then open any notebook in VS Code or Jupyter and run the cells from top to bottom. Notebooks 2 and 3 depend on trained weights (`pinn_weights.pth`, `detector_weights.pth`) produced by earlier notebooks, which are included in the repository.

## Results

| Approach | Average Position Error | Improvement over Physics-Only |
|---|---|---|
| PINN-only (physics alone) | 1.1252 | — |
| Simple blend (Phase 2) | 0.3379 | ~70% |
| Extended Kalman Filter (Phase 3) | 0.0113 | ~99% |

## Future Work

- Extend to 3D orbits, including full camera projection geometry.
- Replace synthetic telescope images with real satellite imagery datasets (e.g., Stanford's SPEED/SPARK pose-estimation datasets) to validate the vision model against real-world data.
- Extend the disturbance model beyond atmospheric drag (e.g., solar radiation pressure, third-body perturbations).

## Status

- ✅ Phase 1 complete
- ✅ Phase 2 complete
- ✅ Phase 3 complete
- 🚧 3D orbit extension planned