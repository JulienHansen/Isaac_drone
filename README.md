# Isaac_drone

[![IsaacSim](https://img.shields.io/badge/IsaacSim-2025.1-blue)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v1.4.0-green)](https://isaac-sim.github.io/IsaacLab/v1.4.0/index.html)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

## Overview

Isaac_drone is a project focused on drone control using NVIDIA's Isaac Lab framework. This repository is structured into various environments, each representing a different phase or aspect of the project. Below is an overview of these environments 

## Environments

### 1. Simulation Environment

**Location:** `source/Isaac_drone/Isaac_drone/tasks`

**Description:** The tasks folder contains every different environment developped during my master thesis, each environment is contained in its own folder, a description of each folder can be view below.

**Key Components:**
- **Phase 1 (Getting to a desired position (fully observable)):** The first environment is provided inside the IsaacLab framework, the goal for the quadcopter is to get to a desired position.
- **Phase 2 (Getting to a desired position (partially observable)):** The second environment is a more complex generalization of the first task, in which the drone has now a camera (or a lidar) and needs to naviguate in a complex environment to get to the desired position, see video below:
- **Phase 3 (Create a formation (fully observable)):** The third environment is a multi-agent tasks, in which 6 drones have to create an Hexagonal formation, see video : 
- **Phase 4 (Adversarial combat):** 


