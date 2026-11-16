"""Kaggle GPU training infrastructure utilities for SentinelGraph.

This package provides reusable, independently testable utilities for setting up
and verifying the Kaggle GPU training environment:

- Environment detection (Python, PyTorch, CUDA, GPU, PyTorch Geometric)
- CUDA tensor execution test
- Minimal graph operation on the available accelerator
- Reproducibility / deterministic settings
- Checkpoint save and restore
- Experiment metadata export (JSON)

Design principles:
- Never assume CUDA or PyTorch versions; detect at runtime.
- Avoid unnecessary package reinstallations.
- Never store credentials.
- Never assume a GPU is always available (CPU fallback supported).
- Persist outputs under /kaggle/working/ (Kaggle's writable output directory).
"""
