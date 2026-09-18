import os

import pytest

from textgraphicalizer import TextGraphicalizer


@pytest.mark.model
def test_real_laya_model_smoke():
    if os.environ.get("TEXTGRAPHICALIZER_RUN_MODEL_TESTS") != "1":
        pytest.skip("set TEXTGRAPHICALIZER_RUN_MODEL_TESTS=1 to download the model")
    pytest.importorskip("laya")
    ontology = {
        "version": 1,
        "concepts": [
            {"id": "rain", "label": "Rain", "description": "Water falling from clouds."},
        ],
        "relations": [
            {"id": "supports", "label": "supports", "description": "Supports the target concept."},
        ],
    }
    graph = TextGraphicalizer(ontology).load_model().fit_transform(
        "Rain fell throughout the afternoon."
    )
    assert graph.is_directed()
    assert graph.graph["model_id"] == "convaiinnovations/laya"
