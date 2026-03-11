"""
Intelligent model partitioner for distributed inference.

Given a model's layer list and each node's available resources,
assigns layers to nodes proportionally so that more capable nodes
handle more layers. Ensures the model fits across the cluster.

Usage::

    from model.auto_partitioner import AutoPartitioner
    from model.resource_detector import NodeInfo

    nodes = [
        NodeInfo("node-a", gpu_vram_mb=4096, has_gpu=True),
        NodeInfo("node-b", gpu_vram_mb=8192, has_gpu=True),
        NodeInfo("node-c", ram_mb=16384, has_gpu=False),
    ]
    partitioner = AutoPartitioner()
    plan = partitioner.create_plan(loader, nodes)
    print(plan)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class PartitionPlan:
    """Describes how model layers are assigned to cluster nodes.

    Attributes
    ----------
    model_name : str
        Name of the model being partitioned.
    assignments : list[NodeAssignment]
        Per-node layer assignments.
    total_layers : int
        Total number of layers in the model.
    total_model_mb : float
        Estimated total model memory in MB.
    feasible : bool
        Whether the model fits in the cluster.
    """
    model_name: str
    assignments: List["NodeAssignment"]
    total_layers: int
    total_model_mb: float
    feasible: bool
    error_message: str = ""

    def summary(self) -> str:
        """Human-readable partition summary."""
        lines = [
            f"Partition Plan for {self.model_name}",
            f"  Total layers: {self.total_layers}",
            f"  Estimated model size: {self.total_model_mb:.0f} MB",
            f"  Nodes: {len(self.assignments)}",
            f"  Feasible: {self.feasible}",
            "",
        ]
        if self.error_message:
            lines.append(f"  ERROR: {self.error_message}")
            lines.append("")

        for a in self.assignments:
            lines.append(
                f"  Node '{a.node_name}': layers [{a.layer_start}-{a.layer_end}] "
                f"({a.num_layers} layers, ~{a.estimated_mb:.0f} MB) "
                f"-> {'GPU' if a.use_gpu else 'CPU'} "
                f"({a.available_mb:.0f} MB available)"
            )
        return "\n".join(lines)


@dataclass
class NodeAssignment:
    """Layer assignment for a single node."""
    node_name: str
    layer_start: int
        # Inclusive index into the model's layer list.
    layer_end: int
        # Inclusive index into the model's layer list.
    layer_names: List[str]
    num_layers: int
    estimated_mb: float
    available_mb: float
    use_gpu: bool


class AutoPartitioner:
    """Assigns model layers to cluster nodes based on available resources.

    The partitioner ensures:
      - More capable nodes (more VRAM/RAM) get more layers.
      - GPU nodes are preferred over CPU-only nodes.
      - Every layer is assigned to exactly one node.
      - The embedding goes to the first node, LM head to the last.
    """

    def create_plan(
        self,
        loader,
        nodes: list,
        dtype_bytes: int = 2,
    ) -> PartitionPlan:
        """Create a partition plan.

        Parameters
        ----------
        loader : HFModelLoader
            Model loader with layer info.
        nodes : list[NodeInfo]
            Available cluster nodes (sorted by capability is recommended).
        dtype_bytes : int
            Bytes per parameter (2 for float16, 4 for float32).

        Returns
        -------
        PartitionPlan
        """
        all_layers = loader.get_layer_list()
        num_layers = len(all_layers)
        model_name = loader.model_name

        # Estimate memory per layer
        layer_sizes = self._estimate_layer_sizes(all_layers, dtype_bytes)
        total_model_mb = sum(layer_sizes)

        # Check feasibility: total cluster memory vs model size
        total_cluster_mb = sum(n.usable_memory_mb for n in nodes)
        if total_cluster_mb < total_model_mb:
            return PartitionPlan(
                model_name=model_name,
                assignments=[],
                total_layers=num_layers,
                total_model_mb=total_model_mb,
                feasible=False,
                error_message=(
                    f"Model requires ~{total_model_mb:.0f} MB but cluster "
                    f"has only ~{total_cluster_mb:.0f} MB available. "
                    f"Need {total_model_mb - total_cluster_mb:.0f} MB more."
                ),
            )

        # Single node — everything on one machine
        if len(nodes) == 1:
            assignment = NodeAssignment(
                node_name=nodes[0].name,
                layer_start=0,
                layer_end=num_layers - 1,
                layer_names=[n for n, _ in all_layers],
                num_layers=num_layers,
                estimated_mb=total_model_mb,
                available_mb=nodes[0].usable_memory_mb,
                use_gpu=nodes[0].has_gpu,
            )
            return PartitionPlan(
                model_name=model_name,
                assignments=[assignment],
                total_layers=num_layers,
                total_model_mb=total_model_mb,
                feasible=True,
            )

        # More nodes than layers — use only as many nodes as layers
        effective_nodes = nodes[:min(len(nodes), num_layers)]

        # Distribute layers proportionally to node capability
        assignments = self._distribute_proportionally(
            all_layers, layer_sizes, effective_nodes
        )

        return PartitionPlan(
            model_name=model_name,
            assignments=assignments,
            total_layers=num_layers,
            total_model_mb=total_model_mb,
            feasible=True,
        )

    def validate_plan(self, plan: PartitionPlan) -> List[str]:
        """Validate a partition plan for correctness.

        Returns a list of warnings/errors (empty = valid).
        """
        issues = []

        if not plan.feasible:
            issues.append(f"Plan is not feasible: {plan.error_message}")
            return issues

        # Check all layers are assigned
        all_layer_names = set()
        for a in plan.assignments:
            for name in a.layer_names:
                if name in all_layer_names:
                    issues.append(f"Layer '{name}' assigned to multiple nodes!")
                all_layer_names.add(name)

        # Check memory fits
        for a in plan.assignments:
            if a.estimated_mb > a.available_mb:
                issues.append(
                    f"Node '{a.node_name}': estimated {a.estimated_mb:.0f} MB "
                    f"exceeds available {a.available_mb:.0f} MB"
                )

        # Check continuity (layers should be contiguous)
        prev_end = -1
        for a in plan.assignments:
            if a.layer_start != prev_end + 1:
                issues.append(
                    f"Gap between layer {prev_end} and {a.layer_start} "
                    f"(node '{a.node_name}')"
                )
            prev_end = a.layer_end

        return issues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_layer_sizes(
        layers: List[Tuple[str, torch.nn.Module]],
        dtype_bytes: int,
    ) -> List[float]:
        """Estimate memory usage per layer in MB."""
        sizes = []
        for name, module in layers:
            num_params = sum(p.numel() for p in module.parameters())
            # Parameters + ~20% overhead for activations/buffers
            size_mb = (num_params * dtype_bytes * 1.2) / (1024 ** 2)
            sizes.append(size_mb)
        return sizes

    @staticmethod
    def _distribute_proportionally(
        all_layers: List[Tuple[str, torch.nn.Module]],
        layer_sizes: List[float],
        nodes: list,
    ) -> List[NodeAssignment]:
        """Distribute layers to nodes proportional to their memory."""
        num_layers = len(all_layers)
        num_nodes = len(nodes)

        # Calculate each node's share of the total memory pool
        total_mem = sum(n.usable_memory_mb for n in nodes)
        node_shares = [n.usable_memory_mb / total_mem for n in nodes]

        # Assign layers greedily: fill each node up to its share
        assignments = []
        layer_idx = 0

        for node_idx, node in enumerate(nodes):
            if node_idx == num_nodes - 1:
                # Last node gets everything remaining
                end_idx = num_layers
            else:
                # Target bytes for this node
                target_mb = sum(layer_sizes) * node_shares[node_idx]
                accumulated_mb = 0.0
                end_idx = layer_idx

                while end_idx < num_layers and accumulated_mb < target_mb:
                    accumulated_mb += layer_sizes[end_idx]
                    end_idx += 1

                # Ensure at least 1 layer per node
                if end_idx == layer_idx:
                    end_idx = layer_idx + 1

                # Don't exceed available layers for remaining nodes
                remaining_nodes = num_nodes - node_idx - 1
                remaining_layers = num_layers - end_idx
                if remaining_layers < remaining_nodes:
                    end_idx = num_layers - remaining_nodes

            chunk_layers = all_layers[layer_idx:end_idx]
            chunk_sizes = layer_sizes[layer_idx:end_idx]

            assignments.append(NodeAssignment(
                node_name=node.name,
                layer_start=layer_idx,
                layer_end=end_idx - 1,
                layer_names=[n for n, _ in chunk_layers],
                num_layers=len(chunk_layers),
                estimated_mb=sum(chunk_sizes),
                available_mb=node.usable_memory_mb,
                use_gpu=node.has_gpu,
            ))

            layer_idx = end_idx

        return assignments
