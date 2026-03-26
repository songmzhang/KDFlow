import time
from itertools import chain
from typing import Optional, Tuple, Union

import ray
import numpy as np
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from kdflow.ray.train.teacher_actor import TeacherRayActor
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


class TeacherActorGroup:
    """
    TeacherActorGroup: Manages multiple TeacherRayActor instances for distributed
    teacher forward (prefilling) in knowledge distillation.
    
    Key design: Teacher actors are scheduled on PlacementGroup bundles using
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES + base_gpu_id, matching the
    same pattern as RolloutActorGroup for unified resource management.
    """
    
    def __init__(
        self,
        strategy,
        num_gpus: int,
        num_gpus_per_node: int = 8,
        num_gpus_per_actor: float = 0.2,
        pg: Optional[Union[PlacementGroup, Tuple[PlacementGroup, list, list]]] = None,
    ):
        """
        Initialize TeacherActorGroup.
        
        Args:
            strategy: Training strategy containing configuration args
            num_gpus: Total number of GPUs available (e.g., 8)
            num_gpus_per_node: Number of GPUs per physical node
            num_gpus_per_actor: Ray GPU resources per actor (fractional for co-location)
            pg: 3-tuple (pg, reordered_bundle_indices, reordered_gpu_ids), PlacementGroup, or None
        """
        logger.info("[TeacherActorGroup] Starting initialization...")
        self.teacher_engines = []
        self._worker_actors = []
        self.strategy = strategy
        self.dp_size = strategy.args.kd.teacher_dp_size
        self.tp_size = strategy.args.kd.teacher_tp_size
        self.pp_size = strategy.args.kd.teacher_pp_size
        self.num_gpus_per_node = num_gpus_per_node
        
        # Parse PG info (same pattern as RolloutActorGroup)
        if pg is not None and isinstance(pg, tuple):
            self._pg, self._reordered_bundle_indices, self._reordered_gpu_ids = pg
        elif pg is not None:
            self._pg = pg
            total_gpus = self.dp_size * self.tp_size * self.pp_size
            self._reordered_bundle_indices = list(range(total_gpus))
            self._reordered_gpu_ids = list(range(total_gpus))
        else:
            self._pg = None
            self._reordered_bundle_indices = None
            self._reordered_gpu_ids = None
        
        # Validate configuration
        required_gpus = self.dp_size * self.tp_size * self.pp_size
        if required_gpus > num_gpus:
            raise ValueError(f"Teacher requires {required_gpus} GPUs (dp={self.dp_size} * tp={self.tp_size} * pp={self.pp_size}) "
                           f"but only {num_gpus} GPUs available")
        
        logger.info(f"[TeacherActorGroup] Creating {self.dp_size} actors with tp_size={self.tp_size} * pp_size={self.pp_size}")
        
        self._create_actors(num_gpus_per_actor)
        
        ray.get([actor.ready.remote() for actor in self.teacher_engines])
        
        logger.info(f"[TeacherActorGroup] All {self.dp_size} actors ready.")
    
    def _create_actors(self, num_gpus_per_actor: float):
        """Create Ray remote TeacherRayActor instances with proper GPU binding via PG."""
        env_vars = {
            name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
        }
        
        num_gpu_per_engine = self.tp_size * self.pp_size
        nnodes_per_engine = max(num_gpu_per_engine // self.num_gpus_per_node, 1)
        
        for i in range(self.dp_size):
            if nnodes_per_engine > 1:  # multi-node tp/pp
                dist_init_addr = self._get_dist_init_addr(i, num_gpu_per_engine)
                engine_actors = []
                for node_idx in range(nnodes_per_engine):
                    gpu_offset = i * num_gpu_per_engine + node_idx * self.num_gpus_per_node

                    if self._reordered_gpu_ids is not None:
                        base_gpu_id = int(self._reordered_gpu_ids[gpu_offset])
                    else:
                        base_gpu_id = 0
                    
                    bundle_idx = self._reordered_bundle_indices[gpu_offset] if self._reordered_bundle_indices else gpu_offset
                    
                    options = {
                        "num_cpus": num_gpus_per_actor,
                        "num_gpus": num_gpus_per_actor,
                        "max_concurrency": 2,
                        "runtime_env": {"env_vars": env_vars},
                    }
                    
                    if self._pg is not None:
                        options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                            placement_group=self._pg,
                            placement_group_capture_child_tasks=True,
                            placement_group_bundle_index=bundle_idx,
                        )
                    
                    actor = TeacherRayActor.options(**options).remote(
                        self.strategy, 
                        base_gpu_id=base_gpu_id,
                        nnodes=nnodes_per_engine,
                        node_rank=node_idx,
                        dist_init_addr=dist_init_addr,
                    )
                    engine_actors.append(actor)
                
                self.teacher_engines.append(engine_actors[0])
                self._worker_actors.extend(engine_actors[1:])
            else:
                # Calculate base_gpu_id from PG topology (same as RolloutActorGroup)
                if self._reordered_gpu_ids is not None:
                    base_gpu_id = int(self._reordered_gpu_ids[i * num_gpu_per_engine])
                else:
                    base_gpu_id = (i * num_gpu_per_engine) % self.num_gpus_per_node
                
                logger.info(f"[TeacherActorGroup] Launching actor {i} with base_gpu_id={base_gpu_id}...")
                
                options = {
                    "num_cpus": num_gpus_per_actor,
                    "num_gpus": num_gpus_per_actor,
                    "max_concurrency": 2,
                    "runtime_env": {
                        "env_vars": env_vars,
                    },
                }
                
                # Schedule on PG bundle if available
                if self._pg is not None and self._reordered_bundle_indices is not None:
                    options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                        placement_group=self._pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=self._reordered_bundle_indices[i * num_gpu_per_engine],
                    )
                
                actor = TeacherRayActor.options(**options).remote(self.strategy, base_gpu_id)
                
                self.teacher_engines.append(actor)
            logger.info(f"[TeacherActorGroup] Actor {i} created, waiting for ready...")
    
    def _get_dist_init_addr(self, engine_idx: int, num_gpu_per_engine: int) -> str:
        offset = engine_idx * num_gpu_per_engine
        bundle_idx = self._reordered_bundle_indices[offset] if self._reordered_bundle_indices else offset

        @ray.remote(num_cpus=0, num_gpus=0)
        def _get_node_ip_and_free_port():
            import socket
            ip = ray.util.get_node_ip_address()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
            return ip, port

        ip, port = ray.get(
            _get_node_ip_and_free_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_bundle_index=bundle_idx,
                )
            ).remote()
        )
        return f"{ip}:{port}"

    def forward(self, global_batch):
        """
        Perform forward pass (prefilling) on all teacher actors in parallel.
        Uses token-based load balancing to distribute batches evenly across actors.
        """
        all_data_ref = ray.put(global_batch)
        
        # === Token-based load balancing ===
        # Calculate token count for each micro-batch (sum of non-padding tokens)
        batch_token_counts = []
        for mb in global_batch:
            # attn_mask indicates non-padding positions
            token_count = mb["tea_attn_mask"].sum().item()
            batch_token_counts.append(token_count)
        
        # Assign batches to actors using greedy algorithm: 
        # Always assign next batch to the actor with fewest tokens
        actor_assignments = [[] for _ in range(self.dp_size)]  # batch indices for each actor
        actor_tokens = [0] * self.dp_size  # running token count for each actor
        
        for batch_idx, token_count in enumerate(batch_token_counts):
            # Find actor with minimum tokens so far
            min_actor = min(range(self.dp_size), key=lambda x: actor_tokens[x])
            actor_assignments[min_actor].append(batch_idx)
            actor_tokens[min_actor] += token_count
        
        futures = []
        for i, actor in enumerate(self.teacher_engines):
            batch_indices = actor_assignments[i]
            futures.append(actor.forward.remote(all_data_ref, batch_indices))
        
        # Use ray.wait to get results as they complete, measuring each actor's timing
        pending = list(futures)
        raw_results = [None] * len(futures)
        future_to_idx = {f: i for i, f in enumerate(futures)}
        
        fwd_start = time.time()
        wait_count = 0
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=120)
            wait_count += 1
            if not ready:
                elapsed = time.time() - fwd_start
                logger.warning(f"[TeacherActorGroup.forward] ray.wait timeout #{wait_count}, "
                               f"elapsed={elapsed:.1f}s, pending_actors={len(pending)}, "
                               f"pending_indices={[future_to_idx[f] for f in pending]}")
                continue
            for ref in ready:
                idx = future_to_idx[ref]
                raw_results[idx] = ray.get(ref)
        
        # Flatten results (extract (batch_idx, batch) tuples from all actors)
        # Each actor returns (results_with_indices, timestamp) where results_with_indices is [(batch_idx, batch), ...]
        indexed_results = list(chain.from_iterable(r for r in raw_results))
        
        # Sort by original batch index to restore the original order
        indexed_results.sort(key=lambda x: x[0])
        results = [batch for _, batch in indexed_results]
        
        return results
    
    def sleep(self):
        """Release GPU memory on all teacher engines."""
        ray.get([actor.sleep.remote() for actor in self.teacher_engines])
    
    def wakeup(self):
        """Resume GPU memory on all teacher engines."""
        ray.get([actor.wakeup.remote() for actor in self.teacher_engines])
        
    def shutdown(self):
        """Shutdown all teacher engines."""
        ray.get([actor.shutdown.remote() for actor in self.teacher_engines])
        logger.info("[TeacherActorGroup] All teacher actors shutdown complete.")