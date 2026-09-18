import networkx as nx
import pytest

from textgraphicalizer import GraphOptimizationError
from textgraphicalizer.optimizer import EdgeEvidence, NodeEvidence, select_graph


def nodes(*probabilities):
    return [
        NodeEvidence(str(index), str(index), probability, 0.9)
        for index, probability in enumerate(probabilities)
    ]


def edge(source, target, probability=0.9, label="rel"):
    return EdgeEvidence(str(source), str(target), label, probability, 0.8)


def kwargs(**overrides):
    result = {
        "node_threshold": 0.5,
        "edge_threshold": 0.5,
        "connected": False,
        "max_node_degree": None,
    }
    result.update(overrides)
    return result


def test_selects_positive_nodes_and_edges():
    graph = select_graph(nodes(0.9, 0.1), [edge(0, 0)], **kwargs())
    assert set(graph.nodes) == {"0"}
    assert graph.number_of_edges() == 0


def test_enforces_max_total_degree():
    graph = select_graph(
        nodes(0.9, 0.9, 0.9),
        [edge(0, 1), edge(0, 2), edge(1, 0)],
        **kwargs(max_node_degree=1),
    )
    assert graph.degree("0") <= 1


def test_connected_graph_includes_all_positive_nodes():
    graph = select_graph(
        nodes(0.9, 0.9, 0.9),
        [edge(0, 1), edge(1, 2)],
        **kwargs(connected=True),
    )
    assert set(graph.nodes) == {"0", "1", "2"}
    assert nx.is_weakly_connected(graph)


def test_connected_graph_can_select_below_threshold_node_for_strong_edge():
    graph = select_graph(
        nodes(0.9, 0.4),
        [edge(0, 1, probability=0.99)],
        **kwargs(connected=True),
    )
    assert set(graph.nodes) == {"0", "1"}
    assert graph.has_edge("0", "1")


def test_connected_graph_reports_infeasibility():
    with pytest.raises(GraphOptimizationError):
        select_graph([], [], **kwargs(connected=True))


def test_empty_candidates_return_empty_graph_when_disconnected():
    graph = select_graph([], [], **kwargs())
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 0
