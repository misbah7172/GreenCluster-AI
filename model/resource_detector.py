"""
Node resource scanner for Kubernetes clusters.

Queries each Kubernetes node for available GPU VRAM, system RAM, CPU
cores, and GPU type. Used by the auto-partitioner to decide how to
distribute model layers across cluster nodes.

Works in two modes:
  1. **Kubernetes mode** — queries the K8s API for real node resources.
  2. **Local mode** — scans the local machine using NVML and psutil.

Usage::

    from model.resource_detector import ResourceDetector

    # Scan Kubernetes cluster
    detector = ResourceDetector(mode="kubernetes")
    nodes = detector.scan()

    # Scan local machine
    detector = ResourceDetector(mode="local")
    nodes = detector.scan()

    for node in nodes:
        print(node)
"""

import logging
import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Resource information for a single cluster node.

    Attributes
    ----------
    name : str
        Node name or hostname.
    gpu_vram_mb : float
        Available GPU VRAM in MB (0 if no GPU).
    gpu_type : str
        GPU model name (e.g., "NVIDIA GeForce RTX 3050 Ti").
    ram_mb : float
        Available system RAM in MB.
    cpu_cores : int
        Number of CPU cores.
    has_gpu : bool
        Whether a GPU is available.
    """
    name: str
    gpu_vram_mb: float = 0.0
    gpu_type: str = "none"
    ram_mb: float = 0.0
    cpu_cores: int = 1
    has_gpu: bool = False
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def usable_memory_mb(self) -> float:
        """Memory available for model layers (GPU VRAM if available, else RAM)."""
        if self.has_gpu and self.gpu_vram_mb > 0:
            # Reserve ~500MB for CUDA overhead
            return max(0, self.gpu_vram_mb - 500)
        # CPU-only: use 70% of RAM (leave room for OS and other processes)
        return self.ram_mb * 0.7


class ResourceDetector:
    """Detects available resources on cluster nodes.

    Parameters
    ----------
    mode : str
        ``"local"`` for local machine, ``"kubernetes"`` for K8s cluster.
    namespace : str
        Kubernetes namespace (only used in kubernetes mode).
    """

    def __init__(self, mode: str = "local", namespace: str = "kai"):
        if mode not in ("local", "kubernetes"):
            raise ValueError(f"mode must be 'local' or 'kubernetes', got '{mode}'")
        self.mode = mode
        self.namespace = namespace

    def scan(self) -> List[NodeInfo]:
        """Scan for available resources.

        Returns
        -------
        list[NodeInfo]
            Sorted by ``usable_memory_mb`` descending (most capable first).
        """
        if self.mode == "local":
            nodes = [self._scan_local()]
        else:
            nodes = self._scan_kubernetes()

        nodes.sort(key=lambda n: n.usable_memory_mb, reverse=True)
        logger.info("Detected %d nodes:", len(nodes))
        for n in nodes:
            logger.info(
                "  %s: GPU=%s (%.0f MB), RAM=%.0f MB, CPU=%d cores",
                n.name, n.gpu_type, n.gpu_vram_mb, n.ram_mb, n.cpu_cores,
            )
        return nodes

    def scan_summary(self) -> Dict:
        """Return a summary of cluster resources.

        Returns
        -------
        dict
            Total GPU VRAM, total RAM, number of nodes, etc.
        """
        nodes = self.scan()
        return {
            "num_nodes": len(nodes),
            "total_gpu_vram_mb": sum(n.gpu_vram_mb for n in nodes),
            "total_ram_mb": sum(n.ram_mb for n in nodes),
            "total_cpu_cores": sum(n.cpu_cores for n in nodes),
            "gpu_nodes": sum(1 for n in nodes if n.has_gpu),
            "cpu_only_nodes": sum(1 for n in nodes if not n.has_gpu),
            "total_usable_mb": sum(n.usable_memory_mb for n in nodes),
            "nodes": [
                {
                    "name": n.name,
                    "gpu_type": n.gpu_type,
                    "gpu_vram_mb": n.gpu_vram_mb,
                    "ram_mb": n.ram_mb,
                    "usable_mb": n.usable_memory_mb,
                }
                for n in nodes
            ],
        }

    # ------------------------------------------------------------------
    # Local scanning
    # ------------------------------------------------------------------

    def _scan_local(self) -> NodeInfo:
        """Scan the local machine."""
        import psutil

        hostname = platform.node()
        ram_mb = psutil.virtual_memory().total / (1024 ** 2)
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

        gpu_vram_mb = 0.0
        gpu_type = "none"
        has_gpu = False

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_vram_mb = mem_info.free / (1024 ** 2)
            gpu_type = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_type, bytes):
                gpu_type = gpu_type.decode("utf-8")
            has_gpu = True
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.info("No GPU detected locally: %s", e)

        return NodeInfo(
            name=hostname,
            gpu_vram_mb=gpu_vram_mb,
            gpu_type=gpu_type,
            ram_mb=ram_mb,
            cpu_cores=cpu_cores,
            has_gpu=has_gpu,
        )

    # ------------------------------------------------------------------
    # Kubernetes scanning
    # ------------------------------------------------------------------

    def _scan_kubernetes(self) -> List[NodeInfo]:
        """Scan Kubernetes cluster nodes."""
        try:
            from kubernetes import client, config as k8s_config
        except ImportError:
            raise ImportError(
                "kubernetes package required. Install with: pip install kubernetes"
            )

        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()

        v1 = client.CoreV1Api()
        k8s_nodes = v1.list_node().items
        nodes = []

        for k8s_node in k8s_nodes:
            name = k8s_node.metadata.name
            labels = k8s_node.metadata.labels or {}

            # Get allocatable resources
            allocatable = k8s_node.status.allocatable or {}
            capacity = k8s_node.status.capacity or {}

            cpu_str = allocatable.get("cpu", capacity.get("cpu", "1"))
            cpu_cores = self._parse_cpu(cpu_str)

            ram_str = allocatable.get("memory", capacity.get("memory", "0"))
            ram_mb = self._parse_memory_to_mb(ram_str)

            # GPU detection via nvidia.com/gpu resource
            gpu_count_str = allocatable.get(
                "nvidia.com/gpu",
                capacity.get("nvidia.com/gpu", "0"),
            )
            gpu_count = int(gpu_count_str)
            has_gpu = gpu_count > 0

            # GPU details from labels (common in GPU-enabled clusters)
            gpu_type = labels.get(
                "nvidia.com/gpu.product",
                labels.get("accelerator", "unknown" if has_gpu else "none"),
            )

            # Estimate VRAM from common GPU types if label available
            gpu_vram_mb = 0.0
            if has_gpu:
                gpu_vram_mb = self._estimate_vram_from_label(gpu_type) * gpu_count

            nodes.append(NodeInfo(
                name=name,
                gpu_vram_mb=gpu_vram_mb,
                gpu_type=gpu_type,
                ram_mb=ram_mb,
                cpu_cores=cpu_cores,
                has_gpu=has_gpu,
                labels=labels,
            ))

        return nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cpu(cpu_str: str) -> int:
        """Parse K8s CPU string (e.g., '4', '2000m') to core count."""
        if cpu_str.endswith("m"):
            return max(1, int(cpu_str[:-1]) // 1000)
        return int(cpu_str)

    @staticmethod
    def _parse_memory_to_mb(mem_str: str) -> float:
        """Parse K8s memory string (e.g., '8Gi', '4096Mi', '4294967296') to MB."""
        mem_str = mem_str.strip()
        if mem_str.endswith("Ki"):
            return float(mem_str[:-2]) / 1024
        if mem_str.endswith("Mi"):
            return float(mem_str[:-2])
        if mem_str.endswith("Gi"):
            return float(mem_str[:-2]) * 1024
        if mem_str.endswith("Ti"):
            return float(mem_str[:-2]) * 1024 * 1024
        # Plain bytes
        try:
            return float(mem_str) / (1024 ** 2)
        except ValueError:
            return 0.0

    @staticmethod
    def _estimate_vram_from_label(gpu_label: str) -> float:
        """Estimate VRAM in MB from GPU product label."""
        gpu_label = gpu_label.lower()
        # Common GPU VRAM sizes
        vram_map = {
            "a100": 81920,     # 80 GB
            "a6000": 49152,    # 48 GB
            "a5000": 24576,    # 24 GB
            "a4000": 16384,    # 16 GB
            "v100": 16384,     # 16 GB
            "t4": 16384,       # 16 GB
            "rtx 4090": 24576, # 24 GB
            "rtx 4080": 16384, # 16 GB
            "rtx 4070": 12288, # 12 GB
            "rtx 4060": 8192,  # 8 GB
            "rtx 3090": 24576, # 24 GB
            "rtx 3080": 10240, # 10 GB
            "rtx 3070": 8192,  # 8 GB
            "rtx 3060": 12288, # 12 GB
            "rtx 3050": 4096,  # 4 GB
            "gtx 1080": 8192,  # 8 GB
            "gtx 1070": 8192,  # 8 GB
            "gtx 1060": 6144,  # 6 GB
            "gtx 1050": 4096,  # 4 GB
            "mx150": 2048,     # 2 GB
            "mx250": 2048,     # 2 GB
        }
        for key, vram in vram_map.items():
            if key in gpu_label:
                return float(vram)
        # Default assumption for unknown GPUs
        return 4096.0
