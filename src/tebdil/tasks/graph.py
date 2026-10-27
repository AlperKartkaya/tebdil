"""Small weighted graph shortest-path and reachability task generator with Dijkstra solver."""

from __future__ import annotations

import heapq
import random
from typing import Dict, List, Tuple

from tebdil.tasks.validators import TaskProblem


_SPLIT_SEED_OFFSETS = {
    "train": 3000,
    "pilot": 7000,
    "validation": 14000,
    "test": 24000,
}


def dijkstra_shortest_path(
    num_nodes: int,
    edges: List[Tuple[int, int, int]],
    source: int,
    target: int,
) -> int:
    """Compute exact shortest path distance from source to target, or -1 if unreachable."""
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = {i: float("inf") for i in range(num_nodes)}
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            return int(d)
        if d > dist[u]:
            continue
        for nxt, w in adj[u]:
            if d + w < dist[nxt]:
                dist[nxt] = d + w
                heapq.heappush(pq, (d + w, nxt))
    return -1


def generate_graph_problems(
    split: str = "pilot",
    num_problems: int = 20,
    seed: int = 42,
) -> List[TaskProblem]:
    """Generate directed weighted shortest-path problems with exact Dijkstra gold labels."""
    offset = _SPLIT_SEED_OFFSETS.get(split, 34000)
    rng = random.Random(seed + offset)
    problems: List[TaskProblem] = []

    for idx in range(num_problems):
        pid = f"graph_{split}_{idx:03d}"
        if split != "test":
            # 4-node DAG (0 -> 1, 2 -> 3) with competing 2-hop vs 3-hop paths
            w01 = rng.randint(2, 9)
            w13 = rng.randint(5, 14)
            w02 = rng.randint(3, 10)
            w23 = rng.randint(4, 12)
            w12 = rng.randint(1, 5)
            edges = [
                (0, 1, w01),
                (1, 3, w13),
                (0, 2, w02),
                (2, 3, w23),
                (1, 2, w12),
            ]
            ans = dijkstra_shortest_path(4, edges, 0, 3)
            direct_paths = [w01 + w13, w02 + w23, w01 + w12 + w23]
            distractors = [str(ans)]
            for dp in direct_paths + [ans + 1, ans + 2]:
                if str(dp) not in distractors:
                    distractors.append(str(dp))
            edge_desc = ", ".join(f"{u}->{v} (cost {w})" for u, v, w in edges)
            prompt = (
                f"Directed graph edges: {edge_desc}. "
                f"What is the minimum path cost from node 0 to node 3? "
                f"Output only the integer cost.\nAnswer: "
            )
            problems.append(
                TaskProblem(
                    problem_id=pid,
                    family="graph",
                    split=split,
                    template_id="graph_4node_diamond_v1",
                    difficulty="medium",
                    prompt=prompt,
                    gold_answer=str(ans),
                    candidate_answers=tuple(distractors[:4]),
                    metadata={"num_nodes": 4, "edges": edges, "source": 0, "target": 3},
                )
            )
        else:
            # 5-node directed graph for held-out test split
            w01 = rng.randint(2, 8)
            w02 = rng.randint(4, 11)
            w13 = rng.randint(3, 9)
            w23 = rng.randint(2, 7)
            w14 = rng.randint(8, 16)
            w34 = rng.randint(2, 8)
            edges = [
                (0, 1, w01),
                (0, 2, w02),
                (1, 3, w13),
                (2, 3, w23),
                (1, 4, w14),
                (3, 4, w34),
            ]
            ans = dijkstra_shortest_path(5, edges, 0, 4)
            alt_costs = [w01 + w14, w01 + w13 + w34, w02 + w23 + w34, ans + 2]
            distractors = [str(ans)]
            for c in alt_costs:
                if str(c) not in distractors:
                    distractors.append(str(c))
            edge_desc = ", ".join(f"{u}->{v}:{w}" for u, v, w in edges)
            prompt = (
                f"Given weighted directed edges [{edge_desc}], find the shortest path distance "
                f"from node 0 to node 4. Output only the integer distance.\nAnswer: "
            )
            problems.append(
                TaskProblem(
                    problem_id=pid,
                    family="graph",
                    split=split,
                    template_id="graph_5node_test",
                    difficulty="hard",
                    prompt=prompt,
                    gold_answer=str(ans),
                    candidate_answers=tuple(distractors[:4]),
                    metadata={"num_nodes": 5, "edges": edges, "source": 0, "target": 4},
                )
            )

    return problems
