"""MILP-based selection of a weighted directed graph."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .errors import GraphOptimizationError


@dataclass(frozen=True)
class NodeEvidence:
    concept_id: str
    label: str
    probability: float
    confidence: float | None = None


@dataclass(frozen=True)
class EdgeEvidence:
    source: str
    target: str
    label: str
    probability: float
    confidence: float | None = None


def _logit(probability: float) -> float:
    epsilon = 1e-6
    p = min(max(float(probability), epsilon), 1.0 - epsilon)
    return log(p / (1.0 - p))


def _add_row(rows: list[dict[int, float]], lower: list[float], upper: list[float], values: dict[int, float], lo: float, hi: float) -> None:
    rows.append(values)
    lower.append(lo)
    upper.append(hi)


def select_graph(
    nodes: Sequence[NodeEvidence],
    edges: Sequence[EdgeEvidence],
    *,
    node_threshold: float,
    edge_threshold: float,
    connected: bool,
    max_node_degree: int | None,
) -> nx.DiGraph:
    """Select a graph by maximizing threshold-relative evidence."""
    graph = nx.DiGraph()
    if not nodes:
        if connected:
            raise GraphOptimizationError("A connected graph requires at least one candidate node")
        return graph

    node_index = {node.concept_id: index for index, node in enumerate(nodes)}
    valid_edges = [
        edge for edge in edges
        if edge.source in node_index and edge.target in node_index and edge.source != edge.target
    ]
    n_nodes = len(nodes)
    n_edges = len(valid_edges)
    flow_count = 2 * n_edges if connected else 0
    variable_count = n_nodes + n_edges + flow_count
    objective = np.zeros(variable_count, dtype=float)
    objective[:n_nodes] = [
        -(_logit(node.probability) - _logit(node_threshold)) for node in nodes
    ]
    objective[n_nodes:n_nodes + n_edges] = [
        -(_logit(edge.probability) - _logit(edge_threshold)) for edge in valid_edges
    ]

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    if connected:
        upper_bounds[n_nodes + n_edges:] = max(1, n_nodes - 1)
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:n_nodes + n_edges] = 1

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    edge_start = n_nodes

    for edge_offset, edge in enumerate(valid_edges):
        edge_var = edge_start + edge_offset
        source_var = node_index[edge.source]
        target_var = node_index[edge.target]
        _add_row(rows, lower, upper, {edge_var: 1.0, source_var: -1.0}, -np.inf, 0.0)
        _add_row(rows, lower, upper, {edge_var: 1.0, target_var: -1.0}, -np.inf, 0.0)

    if max_node_degree is not None:
        for concept_id, concept_var in node_index.items():
            values: dict[int, float] = {concept_var: -float(max_node_degree)}
            for edge_offset, edge in enumerate(valid_edges):
                if edge.source == concept_id or edge.target == concept_id:
                    values[edge_start + edge_offset] = values.get(edge_start + edge_offset, 0.0) + 1.0
            _add_row(rows, lower, upper, values, -np.inf, 0.0)

    if connected:
        root = max(range(n_nodes), key=lambda index: nodes[index].probability)
        _add_row(rows, lower, upper, {root: 1.0}, 1.0, 1.0)

        flow_start = n_nodes + n_edges
        for edge_offset in range(n_edges):
            edge_var = edge_start + edge_offset
            forward_flow = flow_start + 2 * edge_offset
            reverse_flow = forward_flow + 1
            _add_row(rows, lower, upper, {forward_flow: 1.0, edge_var: -(n_nodes - 1)}, -np.inf, 0.0)
            _add_row(rows, lower, upper, {reverse_flow: 1.0, edge_var: -(n_nodes - 1)}, -np.inf, 0.0)

        for node_index_value, concept_id in enumerate(node_index):
            values: dict[int, float] = {}
            for edge_offset, edge in enumerate(valid_edges):
                forward_flow = flow_start + 2 * edge_offset
                reverse_flow = forward_flow + 1
                if edge.source == concept_id:
                    values[forward_flow] = values.get(forward_flow, 0.0) + 1.0
                    values[reverse_flow] = values.get(reverse_flow, 0.0) - 1.0
                elif edge.target == concept_id:
                    values[forward_flow] = values.get(forward_flow, 0.0) - 1.0
                    values[reverse_flow] = values.get(reverse_flow, 0.0) + 1.0
            if node_index_value == root:
                for candidate in range(n_nodes):
                    if candidate != root:
                        values[candidate] = values.get(candidate, 0.0) - 1.0
                _add_row(rows, lower, upper, values, 0.0, 0.0)
            else:
                # The accumulated terms above are outflow-inflow. A selected
                # non-root node must instead have inflow-outflow equal to one.
                values = {column: -coefficient for column, coefficient in values.items()}
                values[node_index_value] = values.get(node_index_value, 0.0) - 1.0
                _add_row(rows, lower, upper, values, 0.0, 0.0)

    if rows:
        matrix = coo_matrix(
            (
                [coefficient for row in rows for coefficient in row.values()],
                (
                    [row_index for row_index, row in enumerate(rows) for _ in row],
                    [column for row in rows for column in row],
                ),
            ),
            shape=(len(rows), variable_count),
        ).tocsr()
        constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    else:
        constraints = ()

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        reason = getattr(result, "message", "unknown solver error")
        raise GraphOptimizationError(f"Could not construct a feasible graph: {reason}")

    selected_nodes = {
        node.concept_id
        for index, node in enumerate(nodes)
        if result.x[index] >= 0.5
    }
    graph.add_nodes_from(
        (
            node.concept_id,
            {
                "label": node.label,
                "probability": node.probability,
                **({"confidence": node.confidence} if node.confidence is not None else {}),
            },
        )
        for node in nodes
        if node.concept_id in selected_nodes
    )
    for edge_offset, edge in enumerate(valid_edges):
        if result.x[edge_start + edge_offset] < 0.5:
            continue
        graph.add_edge(
            edge.source,
            edge.target,
            label=edge.label,
            probability=edge.probability,
            **({"confidence": edge.confidence} if edge.confidence is not None else {}),
        )
    return graph
