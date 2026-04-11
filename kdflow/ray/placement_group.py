import ipaddress
import socket
import logging

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def _sort_key(x):
    _, node_id, gpu_id = x
    try:
        ip_parts = list(ipaddress.ip_address(node_id).packed)
    except ValueError:
        try:
            ip_parts = list(ipaddress.ip_address(socket.getaddrinfo(node_id, None)[0][4][0]).packed)
        except (socket.gaierror, TypeError, ValueError):
            ip_parts = [ord(c) for c in node_id]
    return (ip_parts, gpu_id)


def create_placement_group(num_gpus):
    """Create a placement group and return topology-sorted bundle info.

    Returns:
        (pg, reordered_bundle_indices, reordered_gpu_ids)
    """
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    info_actors = [
        InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i,
            )
        ).remote()
        for i in range(num_gpus)
    ]
    gpu_ids = ray.get([a.get_ip_and_gpu_id.remote() for a in info_actors])
    for a in info_actors:
        ray.kill(a)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_gpus)]
    sorted_infos = sorted(bundle_infos, key=_sort_key)

    reordered_bundle_indices = [info[0] for info in sorted_infos]
    reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_infos]

    for i, idx in enumerate(reordered_bundle_indices):
        logger.info(
            f"  bundle {i:4}, actual_index: {idx:4}, "
            f"node: {gpu_ids[idx][0]}, gpu: {gpu_ids[idx][1]}"
        )

    return pg, reordered_bundle_indices, reordered_gpu_ids
