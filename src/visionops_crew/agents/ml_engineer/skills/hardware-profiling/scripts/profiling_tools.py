"""Custom tools for hardware and framework profiling, packaged inside the hardware-profiling skill."""

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run_command(command: list[str], timeout: int = 10) -> str:
    """Run a system command and return its merged stdout and stderr.

    Args:
        command: List of command arguments.
        timeout: Maximum execution time in seconds. Defaults to 10.

    Returns:
        The command output if successful, or an error message if the command failed.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return f"unavailable: {exc}"

    output = (result.stdout or result.stderr or "").strip()
    return output or f"exited with code {result.returncode}"


def check_hardware_specification() -> dict[str, Any]:
    """Inspect local CPU, memory, and operating system platform details.

    Returns:
        JSON-compatible platform, CPU, and memory details gathered from the
        local operating system.
    """
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "python_executable": os.sys.executable,
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
        },
        "memory": {},
    }

    if platform.system() == "Darwin" and shutil.which("sysctl"):
        info["cpu"]["brand"] = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        memsize = _run_command(["sysctl", "-n", "hw.memsize"])
        if memsize.isdigit():
            info["memory"]["total_gb"] = round(int(memsize) / (1024**3), 2)
    elif platform.system() == "Linux":
        if Path("/proc/cpuinfo").exists():
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.strip().startswith("model name"):
                            info["cpu"]["brand"] = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
        if Path("/proc/meminfo").exists():
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.strip().startswith("MemTotal"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                kb = int(parts[1])
                                info["memory"]["total_gb"] = round(kb / (1024**2), 2)
                                break
            except Exception:
                pass

    return json.loads(json.dumps(info, default=str))


def check_framework_availability() -> dict[str, Any]:
    """Inspect availability and versions of major ML and DL frameworks.

    Checks for PyTorch, JAX, scikit-learn, TensorFlow, transformers, timm,
    fiftyone, and ultralytics, as well as basic workspace info.

    Returns:
        JSON-compatible package availability and workspace metadata.
    """
    packages = {}

    libs = [
        ("torch", "torch"),
        ("jax", "jax"),
        ("sklearn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("transformers", "transformers"),
        ("timm", "timm"),
        ("fiftyone", "fiftyone"),
        ("ultralytics", "ultralytics"),
    ]

    for label, import_name in libs:
        try:
            if import_name == "sklearn":
                import sklearn
                packages[label] = {
                    "available": True,
                    "version": sklearn.__version__,
                    "error": None,
                }
            elif import_name == "torch":
                import torch
                packages[label] = {
                    "available": True,
                    "version": torch.__version__,
                    "error": None,
                }
            elif import_name == "jax":
                import jax
                packages[label] = {
                    "available": True,
                    "version": jax.__version__,
                    "error": None,
                }
            elif import_name == "tensorflow":
                import tensorflow as tf
                packages[label] = {
                    "available": True,
                    "version": tf.__version__,
                    "error": None,
                }
            elif import_name == "transformers":
                import transformers
                packages[label] = {
                    "available": True,
                    "version": transformers.__version__,
                    "error": None,
                }
            elif import_name == "timm":
                import timm
                packages[label] = {
                    "available": True,
                    "version": timm.__version__,
                    "error": None,
                }
            elif import_name == "fiftyone":
                import fiftyone
                packages[label] = {
                    "available": True,
                    "version": fiftyone.__version__,
                    "error": None,
                }
            elif import_name == "ultralytics":
                import ultralytics
                packages[label] = {
                    "available": True,
                    "version": ultralytics.__version__,
                    "error": None,
                }
        except Exception as exc:
            packages[label] = {
                "available": False,
                "version": None,
                "error": str(exc),
            }

    pyproject = Path.cwd() / "pyproject.toml"
    workspace = {
        "cwd": str(Path.cwd()),
        "pyproject_exists": pyproject.exists(),
    }

    info = {
        "packages": packages,
        "workspace": workspace,
    }
    return json.loads(json.dumps(info, default=str))


def check_accelerator_availability() -> dict[str, Any]:
    """Inspect availability of hardware accelerators (e.g., CUDA GPUs, Apple Silicon MPS, TPUs).

    Checks nvidia-smi command, PyTorch CUDA and MPS availability, TensorFlow physical
    devices, and JAX devices.

    Returns:
        JSON-compatible command output and accelerator availability details.
    """
    info = {
        "commands": {},
        "accelerators": {},
    }

    if shutil.which("nvidia-smi"):
        info["commands"]["nvidia_smi"] = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )

    try:
        import torch
        info["accelerators"]["torch_cuda_available"] = torch.cuda.is_available()
        info["accelerators"]["torch_cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["accelerators"]["torch_cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        mps = getattr(torch.backends, "mps", None)
        info["accelerators"]["torch_mps_available"] = bool(
            mps and mps.is_available()
        )
    except Exception:
        pass

    try:
        import jax
        info["accelerators"]["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception:
        pass

    try:
        import tensorflow as tf
        info["accelerators"]["tensorflow_physical_devices"] = [
            str(device) for device in tf.config.list_physical_devices()
        ]
    except Exception:
        pass

    return json.loads(json.dumps(info, default=str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile local hardware and framework environment.")
    parser.add_argument("--hardware", action="store_true", help="Check hardware specifications")
    parser.add_argument("--frameworks", action="store_true", help="Check ML framework availability")
    parser.add_argument("--accelerators", action="store_true", help="Check accelerator availability")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    
    args = parser.parse_args()
    
    # If no flags are provided, run all checks by default
    if not (args.hardware or args.frameworks or args.accelerators):
        args.all = True
        
    results = {}
    if args.hardware or args.all:
        results["hardware"] = check_hardware_specification()
    if args.frameworks or args.all:
        results["frameworks"] = check_framework_availability()
    if args.accelerators or args.all:
        results["accelerators"] = check_accelerator_availability()
        
    print(json.dumps(results, indent=2))
