import logging
import os
import socket
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from kdflow.ray.train.student_actor import StudentRayActor


class StudentActorGroup:
    """
    A group of student actors

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg=None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.duplicate_actors = args.model.ring_attn_size

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        # Parse PG info (same pattern as RolloutActorGroup / TeacherActorGroup)
        if pg is not None and isinstance(pg, tuple):
            self._pg, self._reordered_bundle_indices = pg
        elif pg is not None:
            self._pg = pg
            self._reordered_bundle_indices = None
        else:
            self._pg = None
            self._reordered_bundle_indices = None

        self._initiate_actors(num_gpus_per_actor)

    def _initiate_actors(self, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        pg = self._pg

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

        def _get_bundle_index(rank):
            """Map logical rank to actual PG bundle index using reordered_bundle_indices."""
            if self._reordered_bundle_indices is not None:
                return self._reordered_bundle_indices[rank]
            return rank

        if pg:
            master_actor = StudentRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=_get_bundle_index(0)
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = StudentRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker_actor
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = StudentRayActor.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=_get_bundle_index(rank),
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = StudentRayActor.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(self, *args, **kwargs):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_save_model(self, save_path=None):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote(save_path) for actor in self._actor_handlers]
    
    def async_run_distill(self, data):
        """ Send data to each distill worker and run distillation.
        
        Args: 
            data (list): global batch data
        Returns:
            List[ray.ObjectRef]: List of remote object references to the results
        """
        total_length = len(data)
        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors
        chunk_size = total_length // effective_actors
        refs = []
        for chunk_idx in range(effective_actors):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)
            # Convert Tensors to np.ndarray for Ray zero-copy deserialization
            chunk = [
                {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in mb.items()}
                for mb in data[start_idx:end_idx]
            ]
            chunk_ref = ray.put(chunk)
            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]
                refs.append(actor.fit.remote(chunk_ref))
        return refs
    
    def sleep(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def wakeup(self):
        return ray.get([actor.wakeup.remote() for actor in self._actor_handlers])
    
    def connect_rollout_engines(self, rollout_actors, rollout_tp_size=1):
        """Create Gloo IPC groups between training ranks and rollout engines.

        Following slime's approach: for each engine, create a Gloo group containing
        the training ranks that correspond to the engine's TP workers. This allows
        per-rank CUDA IPC serialization + gather to ensure IPC handles point to
        the same GPU as the receiving sglang TP worker.

        Args:
            rollout_actors: List of RolloutRayActor handles.
            rollout_tp_size: Number of GPUs per rollout engine (TP size).
        """
        refs = [
            actor.connect_rollout_engines.remote(rollout_actors, rollout_tp_size)
            for actor in self._actor_handlers
        ]
        ray.get(refs)

    def update_rollout_weights(self, rollout_actors=None):
        """Stream FSDP weights to rollout engines via Gloo gather + CUDA IPC.

        All ranks participate: each rank serializes its local CUDA IPC data,
        gathers to the source rank via Gloo, and the source rank sends to sglang.
        """
        refs = [
            actor.update_rollout_weights.remote()
            for actor in self._actor_handlers
        ]
        ray.get(refs)
        