---
name: hardware-profiling
description: Inspect hardware specifications, ML framework availability, and accelerator details using scripts.
---

# Hardware Profiling Skill

## Purpose

Use this skill whenever you need to inspect the current execution environment, including CPU, memory, installed ML frameworks (like PyTorch, TensorFlow, JAX), or hardware accelerators (like CUDA GPUs, Apple Silicon MPS, or TPUs).

## Instructions

Step 1: Load the `hardware-profiling` skill to make its scripts available.
Step 2: If general hardware specifications (OS, CPU, RAM) are requested, use the `run_skill_script` tool to execute `scripts/profiling_tools.py` with the `--hardware` argument.
Step 3: If ML frameworks or package versions are requested, use the `run_skill_script` tool to execute `scripts/profiling_tools.py` with the `--frameworks` argument.
Step 4: If GPU, MPS, or TPU accelerators are requested, use the `run_skill_script` tool to execute `scripts/profiling_tools.py` with the `--accelerators` argument.
Step 5: Provide a clear hardware configuration summary to the user.
