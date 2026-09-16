"""Offline tests for SWC tree drawing, type colors, and OBJ mesh parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import viewer


def test_skeleton_tree_uses_parent_child_not_row_order():
    # Nodes listed in an order that would draw a false chord if connected sequentially:
    # 3 is a sibling of 2, not a child of 2.
    skel = pd.DataFrame(
        {
            "rowId": [1, 3, 2, 4],
            "x": [0.0, 1.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, 2.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "link": [-1, 1, 1, 2],
        }
    )
    xs, ys, zs = viewer.skeleton_tree_coords(skel)
    segments = []
    buf = []
    for x, y, z in zip(xs, ys, zs):
        if x is None:
            segments.append(tuple(buf))
            buf = []
        else:
            buf.append((x, y, z))
    # Three real edges: 3-1, 2-1, 4-2. No 3-2 sequential artifact.
    assert len(segments) == 3
    endpoints = {frozenset(seg) for seg in segments}
    assert frozenset([(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]) in endpoints
    assert frozenset([(0.0, 1.0, 0.0), (0.0, 0.0, 0.0)]) in endpoints
    assert frozenset([(0.0, 2.0, 0.0), (0.0, 1.0, 0.0)]) in endpoints


def test_type_colors_are_stable_and_shared():
    mapping = viewer.type_color_map(["DNa02", "DNa01", "DNa02"])
    assert mapping["DNa01"] != mapping["DNa02"]
    assert mapping["DNa01"] == viewer.type_color_map(["DNa02", "DNa01"])["DNa01"]


def test_parse_obj_triangulates_quad():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    xs, ys, zs, i, j, k = viewer.parse_obj_mesh(obj.encode())
    assert xs == [0.0, 1.0, 1.0, 0.0]
    assert len(i) == 2
    assert (i[0], j[0], k[0]) == (0, 1, 2)
    assert (i[1], j[1], k[1]) == (0, 2, 3)


def test_most_synaptic_roi_from_roi_info():
    neurons = pd.DataFrame(
        {
            "bodyId": [1, 2],
            "type": ["DNa01", "DNa02"],
            "roiInfo": [
                json.dumps({"GNG": {"pre": 10, "post": 2}, "AOTU": {"pre": 1, "post": 0}}),
                json.dumps({"GNG": {"pre": 3, "post": 0}, "AOTU": {"pre": 20, "post": 5}}),
            ],
        }
    )
    assert viewer.most_synaptic_roi(neurons) == "AOTU"


def test_build_figure_sets_aspectmode_data():
    fig = viewer.build_figure(
        [
            go.Scatter3d(x=[0, 1, None], y=[0, 0, None], z=[0, 0, None], mode="lines", name="DNa01"),
        ],
        "test",
    )
    assert fig.layout.scene.aspectmode == "data"


def test_load_neurons_from_csv(tmp_path: Path):
    csv_path = tmp_path / "neurons.csv"
    pd.DataFrame({"bodyId": [11, 22], "type": ["DNa01", "DNa02"]}).to_csv(csv_path, index=False)
    df = viewer.load_neurons(csv_path)
    assert list(df["bodyId"]) == [11, 22]


def test_collect_skeleton_traces_skips_failures_and_groups_legend(monkeypatch):
    neurons = pd.DataFrame({"bodyId": [1, 2, 3], "type": ["DNa01", "DNa01", "DNa02"]})

    def fake_fetch(body_id, *, client=None):
        if int(body_id) == 2:
            raise RuntimeError("not proofread")
        if int(body_id) == 3:
            return pd.DataFrame(columns=["rowId", "x", "y", "z", "link"])
        return pd.DataFrame(
            {
                "rowId": [1, 2],
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "link": [-1, 1],
            }
        )

    monkeypatch.setattr(viewer, "fetch_skeletons", fake_fetch)
    traces, skipped = viewer.collect_skeleton_traces(neurons, client=None)
    assert skipped == 2
    assert len(traces) == 1
    assert traces[0].name == "DNa01"
    assert traces[0].legendgroup == "DNa01"


def test_mesh_trace_opacity_range():
    obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    mesh = viewer.mesh_trace(obj, "CNS")
    assert mesh is not None
    assert 0.08 <= mesh.opacity <= 0.15
    assert mesh.name == "ROI CNS"
