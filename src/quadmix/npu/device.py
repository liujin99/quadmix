"""NPU (Ascend NPU) device adaptation layer for QuaDMix.

This module provides device abstraction so that the QuaDMix pipeline
can run on:
  - CPU (development / testing)
  - CUDA (NVIDIA GPU)
  - Ascend NPU (production target)

The paper originally used NVIDIA H100 GPUs for training.
This implementation is designed to be deployable on Ascend NPU
by abstracting device operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"  # Ascend NPU


@dataclass
class NPUDeviceConfig:
    """Configuration for Ascend NPU deployment.

    Ascend NPUs use the CANN (Compute Architecture for Neural Networks)
    toolkit and are accessed through the `torch_npu` package.
    """
    device_id: int = 0
    cann_version: str = "8.0.RC2"
    aoe_mode: bool = False
    memory_limit_mb: Optional[int] = None
    mixed_precision: bool = True
    compile_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": "npu",
            "device_id": self.device_id,
            "cann_version": self.cann_version,
            "aoe_mode": self.aoe_mode,
            "memory_limit_mb": self.memory_limit_mb,
            "mixed_precision": self.mixed_precision,
            "compile_mode": self.compile_mode,
        }


class DeviceManager:
    """Unified device manager for CPU/CUDA/NPU.

    Usage:
        manager = DeviceManager()
        device = manager.get_device()
    """

    def __init__(self, device_type: DeviceType = DeviceType.CPU):
        self.device_type = device_type
        self._device = self._init_device()

    def _init_device(self) -> Any:
        """Initialize the target device. Returns a torch.device-compatible object."""
        if self.device_type == DeviceType.CPU:
            return "cpu"

        elif self.device_type == DeviceType.CUDA:
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.device("cuda:0")
                else:
                    print("[WARN] CUDA requested but not available. Falling back to CPU.")
                    self.device_type = DeviceType.CPU
                    return "cpu"
            except ImportError:
                print("[WARN] PyTorch not installed. Falling back to CPU.")
                self.device_type = DeviceType.CPU
                return "cpu"

        elif self.device_type == DeviceType.NPU:
            try:
                import torch
                import torch_npu  # type: ignore[import-untyped]
                npu_count = torch.npu.device_count()
                if npu_count > 0:
                    return torch.device("npu:0")
                else:
                    print("[WARN] NPU requested but no devices found. Falling back to CPU.")
                    self.device_type = DeviceType.CPU
                    return "cpu"
            except ImportError:
                print(
                    "[WARN] torch_npu not available. "
                    "On the target NPU machine, install CANN + torch_npu. "
                    "See: https://www.hiascend.com/software/cann"
                )
                print("[WARN] Falling back to CPU for local testing.")
                self.device_type = DeviceType.CPU
                return "cpu"

        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

    def get_device(self) -> Any:
        """Get the torch device object."""
        return self._device

    def to_device(self, obj: Any) -> Any:
        """Move a PyTorch module or tensor to the target device."""
        try:
            import torch
            if isinstance(obj, (torch.Tensor, torch.nn.Module)):
                return obj.to(self._device)
            return obj
        except ImportError:
            return obj

    @staticmethod
    def get_npu_launch_command(
        script_path: str,
        num_devices: int = 1,
        extra_args: Optional[str] = None,
    ) -> str:
        """Generate launch command for NPU multi-device training."""
        base = (
            f"RANK_SIZE={num_devices} "
            f"python -m torch.distributed.launch "
            f"--nproc_per_node={num_devices} "
            f"{script_path}"
        )
        if extra_args:
            base += f" {extra_args}"
        return base

    @staticmethod
    def get_npu_info() -> str:
        """Get NPU device info (returns a shell command)."""
        return (
            "# On the NPU machine, run:\n"
            "npu-smi info           # Show NPU device status\n"
            'python -c "import torch; import torch_npu; '
            "print(torch.npu.device_count(), 'devices')\"\n"
            "cat /usr/local/Ascend/CANN_VERSION   # Show CANN version\n"
        )

    def __repr__(self) -> str:
        return f"DeviceManager(device_type={self.device_type.value}, device={self._device})"
