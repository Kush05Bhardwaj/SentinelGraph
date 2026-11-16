"""Environment detection utilities for Kaggle GPU training.

All functions are pure and side-effect free where practical, so they can be
imported and unit-tested outside of a Kaggle notebook.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GpuInfo:
    """Information about a single detected GPU."""

    index: int
    name: str
    memory_total_mb: Optional[int]
    memory_free_mb: Optional[int]
    cuda_capability: Optional[str]


@dataclass
class EnvironmentReport:
    """Aggregated environment detection result."""

    python_version: str
    python_executable: str
    torch_installed: bool
    torch_version: Optional[str]
    torch_cuda_version: Optional[str]
    cuda_available: bool
    device_count: int
    current_device: Optional[int]
    gpus: List[GpuInfo] = field(default_factory=list)
    pyg_installed: bool = False
    pyg_version: Optional[str] = None
    cuda_tensor_test_passed: bool = False
    cuda_tensor_test_error: Optional[str] = None
    pyg_smoke_test_passed: bool = False
    pyg_smoke_test_error: Optional[str] = None
    gpu_graph_op_passed: bool = False
    gpu_graph_op_error: Optional[str] = None
    recommended_action: Optional[str] = None


def detect_python() -> Dict[str, str]:
    """Detect the running Python version and executable path."""
    return {
        "version": sys.version,
        "version_info": ".".join(str(v) for v in sys.version_info[:3]),
        "executable": sys.executable,
    }


def detect_torch() -> Dict[str, Any]:
    """Detect PyTorch installation and its CUDA build version.

    Never assumes a CUDA version; reports whatever torch.version.cuda says
    (None if the CPU build is installed).
    """
    result: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "error": None,
    }
    try:
        import torch  # type: ignore

        result["installed"] = True
        result["version"] = getattr(torch, "__version__", None)
        result["cuda_version"] = getattr(torch, "version", None) and getattr(
            torch.version, "cuda", None
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        result["error"] = str(exc)
    return result


def detect_cuda() -> Dict[str, Any]:
    """Detect CUDA availability and GPU inventory via PyTorch.

    Returns an empty gpu list and cuda_available=False when PyTorch is not
    installed, is a CPU build, or no GPU is attached. This function never
    raises due to missing hardware.
    """
    result: Dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "current_device": None,
        "gpus": [],
        "error": None,
    }
    try:
        import torch  # type: ignore

        result["available"] = bool(torch.cuda.is_available())
        if result["available"]:
            result["device_count"] = int(torch.cuda.device_count())
            result["current_device"] = int(torch.cuda.current_device())
            for i in range(result["device_count"]):
                try:
                    props = torch.cuda.get_device_properties(i)
                    result["gpus"].append(
                        {
                            "index": i,
                            "name": props.name,
                            "memory_total_mb": int(props.total_memory) // (1024 * 1024),
                            "cuda_capability": f"{props.major}.{props.minor}",
                        }
                    )
                except Exception:  # pragma: no cover - hardware dependent
                    result["gpus"].append(
                        {
                            "index": i,
                            "name": "unknown",
                            "memory_total_mb": None,
                            "cuda_capability": None,
                        }
                    )
    except Exception as exc:  # pragma: no cover - depends on environment
        result["error"] = str(exc)
    return result


def detect_pyg() -> Dict[str, Any]:
    """Detect PyTorch Geometric installation without importing heavy deps."""
    result: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "error": None,
    }
    try:
        import torch_geometric  # type: ignore

        result["installed"] = True
        result["version"] = getattr(torch_geometric, "__version__", None)
    except Exception as exc:  # pragma: no cover - depends on environment
        result["error"] = str(exc)
    return result


def test_cuda_tensor() -> Dict[str, Any]:
    """Execute a minimal tensor operation on CUDA if available.

    Returns passed=True only when a tensor can be allocated on CUDA and a
    simple arithmetic op completes. Gracefully returns passed=False when no
    GPU is available.
    """
    result: Dict[str, Any] = {
        "passed": False,
        "error": None,
        "device": None,
    }
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            result["error"] = "CUDA not available"
            result["device"] = "cpu"
            return result
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = x * 2.0
        result["passed"] = bool(torch.allclose(y, torch.tensor([2.0, 4.0, 6.0], device=device)))
        result["device"] = "cuda"
        # Free memory explicitly
        del x, y
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - hardware dependent
        result["error"] = str(exc)
        result["device"] = "cuda"
    return result


def test_pyg_smoke() -> Dict[str, Any]:
    """Smoke-test PyTorch Geometric by building a tiny Data object.

    Does not train; only verifies the import and a trivial graph construction.
    """
    result: Dict[str, Any] = {
        "passed": False,
        "error": None,
    }
    try:
        import torch  # type: ignore
        from torch_geometric.data import Data  # type: ignore

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([[1.0], [2.0]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        result["passed"] = data.num_nodes == 2 and data.num_edges == 2
    except Exception as exc:  # pragma: no cover - depends on environment
        result["error"] = str(exc)
    return result


def test_gpu_graph_operation() -> Dict[str, Any]:
    """Run a minimal graph operation (edge index coalescing) on the accelerator.

    Uses CUDA when available, otherwise CPU. Verifies PyG can execute on the
    selected device without assuming hardware.
    """
    result: Dict[str, Any] = {
        "passed": False,
        "error": None,
        "device": None,
    }
    try:
        import torch  # type: ignore
        from torch_geometric.data import Data  # type: ignore
        from torch_geometric.utils import coalesce  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result["device"] = device.type
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device
        )
        x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float, device=device)
        data = Data(x=x, edge_index=edge_index)
        edge_index_coalesced, _ = coalesce(data.edge_index, None, data.num_nodes, data.num_nodes)
        result["passed"] = int(edge_index_coalesced.shape[1]) > 0
        del data, edge_index, edge_index_coalesced, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - hardware dependent
        result["error"] = str(exc)
        result["device"] = "unknown"
    return result


def build_environment_report() -> EnvironmentReport:
    """Build a full EnvironmentReport by running all detections and tests."""
    py = detect_python()
    torch_info = detect_torch()
    cuda_info = detect_cuda()
    pyg_info = detect_pyg()
    tensor_test = test_cuda_tensor()
    pyg_test = test_pyg_smoke()
    graph_test = test_gpu_graph_operation()

    gpus = [
        GpuInfo(
            index=g["index"],
            name=g["name"],
            memory_total_mb=g["memory_total_mb"],
            memory_free_mb=None,
            cuda_capability=g["cuda_capability"],
        )
        for g in cuda_info["gpus"]
    ]

    recommended = None
    if not torch_info["installed"]:
        recommended = "Install PyTorch (CPU or CUDA build) before training."
    elif not cuda_info["available"]:
        recommended = "CUDA is not available; training will run on CPU."
    elif not pyg_info["installed"]:
        recommended = "Install torch-geometric for GNN training."
    elif not tensor_test["passed"]:
        recommended = "CUDA tensor test failed; verify CUDA/PyTorch compatibility."
    elif not pyg_test["passed"]:
        recommended = "PyG smoke test failed; verify torch-geometric installation."
    elif not graph_test["passed"]:
        recommended = "GPU graph operation failed; verify accelerator compatibility."

    return EnvironmentReport(
        python_version=py["version"],
        python_executable=py["executable"],
        torch_installed=torch_info["installed"],
        torch_version=torch_info["version"],
        torch_cuda_version=torch_info["cuda_version"],
        cuda_available=cuda_info["available"],
        device_count=cuda_info["device_count"],
        current_device=cuda_info["current_device"],
        gpus=gpus,
        pyg_installed=pyg_info["installed"],
        pyg_version=pyg_info["version"],
        cuda_tensor_test_passed=tensor_test["passed"],
        cuda_tensor_test_error=tensor_test["error"],
        pyg_smoke_test_passed=pyg_test["passed"],
        pyg_smoke_test_error=pyg_test["error"],
        gpu_graph_op_passed=graph_test["passed"],
        gpu_graph_op_error=graph_test["error"],
        recommended_action=recommended,
    )


def environment_report_to_dict(report: EnvironmentReport) -> Dict[str, Any]:
    """Serialize an EnvironmentReport to a JSON-friendly dict."""
    return {
        "python_version": report.python_version,
        "python_executable": report.python_executable,
        "torch_installed": report.torch_installed,
        "torch_version": report.torch_version,
        "torch_cuda_version": report.torch_cuda_version,
        "cuda_available": report.cuda_available,
        "device_count": report.device_count,
        "current_device": report.current_device,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "memory_total_mb": g.memory_total_mb,
                "cuda_capability": g.cuda_capability,
            }
            for g in report.gpus
        ],
        "pyg_installed": report.pyg_installed,
        "pyg_version": report.pyg_version,
        "cuda_tensor_test_passed": report.cuda_tensor_test_passed,
        "cuda_tensor_test_error": report.cuda_tensor_test_error,
        "pyg_smoke_test_passed": report.pyg_smoke_test_passed,
        "pyg_smoke_test_error": report.pyg_smoke_test_error,
        "gpu_graph_op_passed": report.gpu_graph_op_passed,
        "gpu_graph_op_error": report.gpu_graph_op_error,
        "recommended_action": report.recommended_action,
    }


def detect_nvidia_smi() -> Dict[str, Any]:
    """Detect driver/CUDA info via nvidia-smi when present.

    Never fails hard; returns an empty/error dict when nvidia-smi is absent.
    """
    result: Dict[str, Any] = {"available": False, "output": None, "error": None}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        result["available"] = True
        result["output"] = output.strip()
    except FileNotFoundError:
        result["error"] = "nvidia-smi not found"
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error"] = str(exc)
    return result


def should_reinstall_torch(report: EnvironmentReport, cuda_major: Optional[int] = None) -> bool:
    """Decide whether a torch reinstall is warranted.

    Heuristic: reinstall only when torch is missing, or when CUDA is available
    but torch reports no CUDA build. Does not perform the install; callers
    decide. Avoids unnecessary reinstalls per SG-004 constraints.
    """
    if not report.torch_installed:
        return True
    if report.cuda_available and report.torch_cuda_version is None:
        return True
    return False
