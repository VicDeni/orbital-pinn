# Vision-Assisted Physics-Informed Neural Network for Satellite Orbit Estimation

A two-part project exploring how physics-informed machine learning and computer vision can work together to predict and correct satellite orbital trajectories.

## Overview

Traditional orbital simulations solve Newton's equations of motion through numerical integration. This project investigates an alternative: a **Physics-Informed Neural Network (PINN)** that learns orbital trajectories by embedding the governing differential equations directly into its training process, rather than learning purely from labeled data.

The project then extends this further with a **vision-assisted correction layer** — simulating how real orbit-tracking systems fuse physics-based predictions with observational data (e.g., from telescopes) to correct for real-world disturbances a pure physics model can't anticipate.

## Project Structure

### Phase 1 — Physics-Informed Neural Network
`notebooks/01_orbital_mechanics_baseline.ipynb`

- Simulates a two-body orbital system using classical numerical methods (`scipy.integrate.solve_ivp`) to generate a trusted ground-truth trajectory.
- Builds a PINN (PyTorch) that predicts satellite position from time alone.
- Trains the network using a physics-informed loss function — via automatic differentiation, the network's own predicted acceleration is checked against Newton's law of gravitation, so it learns to obey the governing equation rather than just memorizing data points.
- Compares the PINN's learned trajectory against the classical numerical solution and evaluates accuracy.

### Phase 2 — Vision-Assisted Orbit Correction *(in progress)*
`notebooks/02_vision_assisted_correction.ipynb`

- Introduces a realistic disturbance (atmospheric drag) so the true orbit no longer matches the PINN's idealized physics.
- Simulates synthetic "telescope images" of the satellite's true position.
- Builds a small computer vision model (CNN) to detect the satellite's position directly from these images.
- Fuses these vision-based observations into the PINN's prediction to correct for the disturbance — demonstrating how combining a physics model with real observations produces a more robust orbit estimate than either alone.

## Key Concepts Demonstrated

- Physics-informed machine learning (PINNs) and automatic differentiation
- Classical numerical ODE solving for orbital dynamics
- Custom physics-based loss functions in PyTorch
- Computer vision for object localization (CNN-based detection)
- Sensor fusion — correcting model predictions with observational data
- Interactive and comparative scientific visualization

## Setup

```bash
git clone https://github.com/VicDeni/orbital-pinn.git
cd orbital-pinn
python3 -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then open either notebook in VS Code or Jupyter and run the cells from top to bottom.

## Status

- Phase 1 complete
- Phase 2 in progress