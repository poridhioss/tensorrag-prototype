from app.models.pipeline import EdgeConfig, NodeConfig, PipelineRequest
from app.services.dag import topological_sort, validate_dag


def _make_pipeline(nodes, edges):
    return PipelineRequest(
        pipeline_id="test-pipe",
        nodes=[NodeConfig(**n) for n in nodes],
        edges=[EdgeConfig(**e) for e in edges],
    )


def test_valid_linear_dag():
    pipeline = _make_pipeline(
        nodes=[
            {"id": "n1", "type": "data_load"},
            {"id": "n2", "type": "data_split"},
        ],
        edges=[
            {"source": "n1", "target": "n2",
             "source_output": "dataset", "target_input": "dataset"},
        ],
    )
    errors = validate_dag(pipeline)
    assert errors == []


def test_cycle_detection():
    pipeline = _make_pipeline(
        nodes=[
            {"id": "n1", "type": "data_load"},
            {"id": "n2", "type": "data_split"},
        ],
        edges=[
            {"source": "n1", "target": "n2",
             "source_output": "dataset", "target_input": "dataset"},
            {"source": "n2", "target": "n1",
             "source_output": "train_dataset", "target_input": "dataset"},
        ],
    )
    errors = validate_dag(pipeline)
    assert any("cycle" in e.lower() for e in errors)


def test_missing_node_reference():
    pipeline = _make_pipeline(
        nodes=[
            {"id": "n1", "type": "data_load"},
        ],
        edges=[
            {"source": "n1", "target": "n_missing",
             "source_output": "dataset", "target_input": "dataset"},
        ],
    )
    errors = validate_dag(pipeline)
    assert any("n_missing" in e for e in errors)


def test_missing_input_edge():
    pipeline = _make_pipeline(
        nodes=[
            {"id": "n1", "type": "data_split"},  # requires input but has no edges
        ],
        edges=[],
    )
    errors = validate_dag(pipeline)
    assert any("requires inputs" in e for e in errors)


def test_topological_sort_linear():
    pipeline = _make_pipeline(
        nodes=[
            {"id": "n1", "type": "data_load"},
            {"id": "n2", "type": "data_split"},
            {"id": "n3", "type": "model_define"},
        ],
        edges=[
            {"source": "n1", "target": "n2",
             "source_output": "dataset", "target_input": "dataset"},
        ],
    )
    levels = topological_sort(pipeline)
    # n1 and n3 have no incoming edges, should be in the first level
    assert "n1" in levels[0]
    assert "n3" in levels[0]
    # n2 depends on n1
    found_n2 = False
    for level in levels[1:]:
        if "n2" in level:
            found_n2 = True
    assert found_n2


def test_topological_sort_diamond():
    pipeline = _make_pipeline(
        nodes=[
            {"id": "a", "type": "data_load"},
            {"id": "b", "type": "data_split"},
            {"id": "c", "type": "model_define"},
            {"id": "d", "type": "train"},
        ],
        edges=[
            {"source": "a", "target": "b",
             "source_output": "dataset", "target_input": "dataset"},
            {"source": "a", "target": "c",
             "source_output": "dataset", "target_input": "default"},
            {"source": "b", "target": "d",
             "source_output": "train_dataset", "target_input": "train_dataset"},
            {"source": "c", "target": "d",
             "source_output": "model_spec", "target_input": "model_spec"},
        ],
    )
    levels = topological_sort(pipeline)
    assert levels[0] == ["a"]
    # b and c should be at the same level
    assert set(levels[1]) == {"b", "c"}
    assert levels[2] == ["d"]
