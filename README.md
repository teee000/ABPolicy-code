# ABPolicy: Asynchronous B-spline Flow Policy for Smooth and Responsive Robotic Manipulation

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://teee000.github.io/ABPolicy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper **"ABPolicy: An Asynchronous B-spline Flow Policy for Smooth and Responsive Robotic Manipulation"**. 

## 📖 Overview

Robotic manipulation requires policies that are temporally smooth and highly responsive to dynamic environments. Conventional synchronous inference in the raw action space often suffers from intra-chunk jitter, inter-chunk discontinuities, and stop-and-go execution. 

**ABPolicy** solves these challenges by operating in a **continuous B-spline control-point action space** and utilizing an **asynchronous flow-matching** framework. By decoupling policy inference from robot execution and optimizing trajectory continuity, ABPolicy delivers real-time, continuous, and jitter-free manipulation.

## ✨ Key Technical Highlights

* **B-spline Control-Point Action Space:** Instead of predicting raw discrete actions, ABPolicy predicts continuous B-spline control points (parameterized as absolute joint angles). This inherently guarantees **intra-chunk smoothness** and eliminates high-frequency jitter.
* **Asynchronous Flow-Matching:** The policy updates in the background while the robot continues to execute, completely eliminating the "stop-and-go" delays caused by synchronous inference.
* **Bidirectional Action Chunking & Continuity Refitting:** We jointly model a short window of past and future actions. When a new chunk is generated asynchronously, we apply a **continuity-constrained refitting optimization** to locally adjust the initial control points, seamlessly stitching trajectories together to ensure strict **inter-chunk continuity**.
* **Robust Visual Features:** We leverage pre-trained **DINOv2** to extract rich, generalizable visual representations from multi-camera setups.
* **Hardware Deployment:** Validated extensively on the **Agilex Piper** 6-DoF robotic arm across 7 tasks, including challenging dynamic scenarios with moving objects.

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/teee000/ABPolicy-code.git](https://github.com/teee000/ABPolicy-code.git)
cd ABPolicy-code
```

**2. Create the Conda environment**
```bash
conda create -n abpolicy python=3.10 -y
conda activate abpolicy
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## 🗄️ Dataset Preparation

The dataset utilizes the `HDF5` format for efficient loading. The `HDF5Dataset` class supports multi-camera setups, `qpos` (proprioception), and `action` (absolute joint angles) loading.

Your dataset tree should look like this:
```text
data/
└── your_task_name/
    ├── episode_0.hdf5
    ├── episode_1.hdf5
    └── ...
```

## 🚀 Usage

### Training

To train the ABPolicy using Flow Matching on your dataset:
```bash
python train.py 
```
*The training pipeline automatically normalizes the state/action space and builds the B-spline control point targets.*

### Asynchronous Inference & Refitting

During inference on the real robot (Agilex Piper), the policy runs asynchronously. The core refitting logic is handled by our `BSplineFitter`:

```python
# Example snippet of the inter-chunk refitting process
from bspline_fitter import BSplineFitter

fitter = BSplineFitter(T=chunk_size, k=3, n_ctrl=8)

# refit_prefix_w ensures continuity with the previously executed actions
# by optimizing the first few control points of the newly predicted chunk
smoothed_control_points = fitter.refit_prefix_w(
    y_prefix=executed_past_actions,
    ctrl_y=predicted_control_points,
    n_prefix=8,
    n_free=4,
    last_pt_weight=0.05
)

# Rebuild the smooth trajectory 
smooth_trajectory, _ = fitter.rebuild(smoothed_control_points)
```

## 🤖 Hardware Setup

The experiments in the paper are conducted using the **Agilex Piper** robotic arm. 
* **Action Space**: Absolute joint angles (6 DoF + Gripper).


## 🤝 Acknowledgments

* **DINOv2**: We utilize DINOv2 for extracting robust visual features.
* **Agilex Piper**: Hardware platform utilized for physical deployments.
