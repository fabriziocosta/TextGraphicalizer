import os
from pathlib import Path

import pytest


@pytest.mark.model
def test_initial_usage_notebook_executes_with_real_model(monkeypatch):
    if os.environ.get("TEXTGRAPHICALIZER_RUN_MODEL_TESTS") != "1":
        pytest.skip("set TEXTGRAPHICALIZER_RUN_MODEL_TESTS=1 to download the model")
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")

    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook_path = Path(__file__).parents[1] / "notebooks" / "initial_usage.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parents[1])}},
    )
    client.execute()

    assert any(
        "graphs = []" in "".join(cell.get("source", []))
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
