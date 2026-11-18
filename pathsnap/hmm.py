"""HMM over candidate arcs and Viterbi decoding."""

from functools import lru_cache

import networkx as nx
import numpy as np


def decode_route(trip, geometry, candidates, probabilities, *, split_s, beta_m=20.0):
    """Decode the most likely arc sequence for each connected portion of a trip."""
    if beta_m <= 0 or split_s <= 0:
        raise ValueError("transition scale and gap limit must be positive")
    points, starts, ends = geometry
    timestamps = trip.track["t"].to_numpy()
    from_nodes, to_nodes = trip.arcs[["from_node", "to_node"]].to_numpy().T
    valid_arcs = np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
    lengths = np.full(len(starts), np.nan)
    lengths[valid_arcs] = np.linalg.norm(ends[valid_arcs] - starts[valid_arcs], axis=1)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(trip.nodes)))
    for start, end, length in zip(from_nodes, to_nodes, lengths):
        if np.isfinite(length) and (not graph.has_edge(start, end) or length < graph[start][end]["weight"]):
            graph.add_edge(start, end, weight=length)

    @lru_cache(maxsize=512)
    def paths_from(node, cutoff):
        return nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight="weight")

    def transition(previous, current, previous_offsets, current_offsets, displacement):
        limit = max(500, 500 * int(np.ceil((displacement + 8 * beta_m) / 500)))
        network = np.array([
            [paths_from(to_nodes[left], limit).get(from_nodes[right], np.inf) for right in current]
            for left in previous
        ])
        network += (lengths[previous] - previous_offsets)[:, None] + current_offsets
        along = current_offsets - previous_offsets[:, None]
        on_same_arc = (previous[:, None] == current) & (along >= 0)
        network = np.where(on_same_arc, along, network)
        return -np.abs(network - displacement) / beta_m

    output = np.full(len(points), -1, dtype=int)
    groups = candidates.reset_index(drop=True).groupby("observation_id", sort=True).indices
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(candidates),) or not np.isfinite(probabilities).all():
        raise ValueError("expected one finite emission probability per candidate")
    emissions = np.log(np.clip(probabilities, 1e-12, 1.0))
    arcs = candidates["arc_id"].to_numpy(dtype=int)
    offsets = candidates["position_along_arc"].to_numpy() * lengths[arcs]
    chain = []
    scores = np.empty(0)

    def finish():
        if chain:
            state = int(scores.argmax())
            for observation, indices, parents in reversed(chain):
                output[observation] = arcs[indices[state]]
                state = int(parents[state])
            chain.clear()

    for observation, indices in groups.items():
        parents = np.full(len(indices), -1, dtype=int)
        if chain:
            last_observation, last_indices, _ = chain[-1]
            elapsed = timestamps[observation] - timestamps[last_observation]
            if observation != last_observation + 1 or not 0 < elapsed <= split_s:
                finish()
            else:
                distance = np.linalg.norm(points[observation] - points[last_observation])
                options = scores[:, None] + transition(
                    arcs[last_indices], arcs[indices], offsets[last_indices], offsets[indices], distance
                )
                parents = options.argmax(axis=0)
                best = options[parents, np.arange(len(indices))]
                if np.isfinite(best).any():
                    scores = emissions[indices] + best
                    chain.append((observation, indices, parents))
                    continue
                finish()
        scores = emissions[indices]
        chain.append((observation, indices, parents))
    finish()
    return output
