from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git_command(project_root: Path, args: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def get_git_metadata(project_root: Path) -> Dict[str, Optional[str]]:
    return {
        "commit": _run_git_command(project_root, ["rev-parse", "HEAD"]),
        "commit_short": _run_git_command(project_root, ["rev-parse", "--short", "HEAD"]),
        "branch": _run_git_command(project_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def get_torch_metadata() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "imported": False,
        "version": None,
        "cuda_available": None,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": None,
        "devices": [],
        "error": None,
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        info["error"] = str(exc)
        return info

    info["imported"] = True
    info["version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version()
    info["device_count"] = torch.cuda.device_count()

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            device_info: Dict[str, Any] = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
            }
            try:
                props = torch.cuda.get_device_properties(index)
                device_info.update(
                    {
                        "total_memory_bytes": int(props.total_memory),
                        "multi_processor_count": int(props.multi_processor_count),
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
            except Exception:
                pass
            devices.append(device_info)
        info["devices"] = devices

    return info


def build_runtime_info(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    status: str,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
        "timestamp_utc": utc_now_iso(),
        "project_root": str(project_root.resolve()),
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "cwd": str(Path.cwd().resolve()),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version,
        },
        "process": {
            "pid": os.getpid(),
            "argv": sys.argv,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        },
        "git": get_git_metadata(project_root),
        "torch": get_torch_metadata(),
    }

    if config is not None:
        payload["config_summary"] = {
            "project_name": config.get("project", {}).get("name"),
            "dataset": config.get("data", {}).get("dataset"),
            "device_requested": config.get("runtime", {}).get("device"),
            "batch_size": config.get("training", {}).get("batch_size"),
            "epochs": config.get("training", {}).get("epochs"),
        }

    if extra:
        payload.update(extra)

    return payload


def write_runtime_info(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
