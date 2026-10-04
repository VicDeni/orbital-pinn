# Vision-Assisted Physics-Informed Neural Network for Satellite Orbit Estimation

A four-part project exploring how physics-informed machine learning, computer vision, and adaptive sensor fusion can work together to predict and correct satellite orbital trajectories — starting in a simplified 2D plane and ending with a full 3D, bearing-only tracking pipeline.

## Overview

Traditional orbital simulations solve Newton's equations of motion through numerical integration. This project investigates an alternative: a **Physics-Informed Neural Network (PINN)** that learns orbital trajectories by embedding the governing differential equations directly into its training process, rather than learning purely from labeled data.

The project then extends this with a **vision-assisted correction layer** simulating how real orbit-tracking systems fuse physics-based predictions with observational data, replaces a naive fixed-ratio correction with an **Extended Kalman Filter**, and finally rebuilds the entire pipeline in **three dimensions** — introducing real camera projection geometry and the depth ambiguity that comes with it.

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

### Phase 4 — 3D Orbit Estimation with Bearing-Only Vision Fusion
`notebooks/04_3d_orbital_mechanics.ipynb`

- Extends the PINN, physics model, and conservation-law constraints from 2D into full 3D, including a genuinely tilted orbital plane and a 3D angular-momentum vector.
- Introduces a real pinhole camera projection model, replacing Phase 2's flat pixel rescaling — and with it, a genuine depth ambiguity: a single 2D image fixes a direction to the satellite, not its distance.
- Trains a CNN detector on synthetic 3D telescope images and uses its pixel predictions as a **bearing-only** measurement.
- Builds an Extended Kalman Filter whose "Extended" behavior is doing real work for the first time in the project: both the gravity-only process model and the nonlinear camera projection measurement model are re-linearized at every time step.
- Achieves a ~95% reduction in average 3D position error compared to the physics-only baseline, using nothing but a direction to the satellite and a simple two-body gravity model.

## Key Concepts Demonstrated

- Physics-informed machine learning (PINNs) and automatic differentiation
- Classical numerical ODE solving for orbital dynamics, in 2D and 3D
- Custom physics-based loss functions in PyTorch, including conservation-law constraints (energy and angular momentum, including the 3D angular-momentum vector)
- Curriculum learning for long-horizon neural ODE solutions
- Computer vision for object localization (CNN-based detection)
- Pinhole camera projection geometry and bearing-only (angles-only) tracking
- Sensor fusion via Extended Kalman Filtering, with both linear and genuinely nonlinear measurement models
- Interactive and comparative scientific visualization, in 2D and 3D

## Setup

```bash
git clone https://github.com/VicDeni/orbital-pinn.git
cd orbital-pinn
python3 -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then open any notebook in VS Code or Jupyter and run the cells from top to bottom. Notebooks 2 and 3 depend on trained weights (`pinn_weights.pth`, `detector_weights.pth`) produced by earlier notebooks; Notebook 4 produces and uses its own separate 3D weights (`pinn_3d_weights.pth`, `detector_3d_weights.pth`). All trained weights are included in the repository.

## Results

| Approach | Average Position Error | Improvement over Physics-Only |
|---|---|---|
| PINN-only, 2D (physics alone) | 1.1252 | — |
| Simple blend, 2D (Phase 2) | 0.3379 | ~70% |
| Extended Kalman Filter, 2D (Phase 3) | 0.0113 | ~99% |
| PINN-only, 3D (physics alone) | 1.2243 | — |
| Extended Kalman Filter, 3D bearing-only (Phase 4) | 0.0660 | ~95% |

## Future Work

- Replace synthetic telescope images with real satellite imagery datasets (e.g., Stanford's SPEED/SPARK pose-estimation datasets) to validate the vision model against real-world data.
- Extend the disturbance model beyond atmospheric drag (e.g., solar radiation pressure, third-body perturbations), including in the 3D pipeline.
- Add a second camera (stereo triangulation) to Phase 4 as a comparison point against the current bearing-only approach.

## Status

- ✅ Phase 1 complete
- ✅ Phase 2 complete
- ✅ Phase 3 complete
- ✅ Phase 4 complete