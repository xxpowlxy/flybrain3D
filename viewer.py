"""Interactive 3D Plotly viewer for a MaleCNS neuron subset.

Modes:
  python viewer.py                 # skeletons only
  python viewer.py --with-mesh     # skeletons + transparent CNS/brain hull
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from fetch_connectome import (
    DATASET,
    MAX_NEURONS,
    SERVER,
    TYPE_REGEX,
    fetch_neurons,
    get_client,
)

# Whole-CNS / whole-brain outline names to try first (male-cns:v1.0).
# If none of these exist as a fetchable ROI mesh, we fall back to the ROI
# that contains the most synapses in the loaded subset.
OUTLINE_ROI_CANDIDATES = (
    "CNS",
    "Brain",
    "brain",
    "hemibrain",
    "Neuropil",
    "neuropil",
    "VNC",
)

MESH_OPACITY = 0.12
MESH_COLOR = "#d0d0d0"


def fetch_skeletons(body_id, *, client=None):
    """Fetch the SWC skeleton for one ``bodyId``.

    neuPrint-python exposes ``fetch_skeleton`` (singular). This wrapper is
    called once per bodyId, matching the per-neuron fetch described in the
    viewer spec.
    """
    from neuprint import fetch_skeleton

    return fetch_skeleton(int(body_id), client=client)


def load_neurons(input_path: Path) -> pd.DataFrame:
    """Load ``neurons.csv`` if present; otherwise query neuPrint with the shared regex."""
    if input_path.is_file():
        df = pd.read_csv(input_path)
        if "bodyId" not in df.columns:
            raise SystemExit(f"{input_path} has no bodyId column.")
        print(f"Loaded {len(df)} neurons from {input_path}")
        return df

    print(
        f"{input_path} not found; fetching neurons with type regex "
        f"{TYPE_REGEX!r} (max {MAX_NEURONS}) from {DATASET}."
    )
    neuron_df, _roi_df = fetch_neurons(type_regex=TYPE_REGEX, max_neurons=MAX_NEURONS)
    if neuron_df is None or neuron_df.empty:
        raise SystemExit(f"No neurons matched {TYPE_REGEX!r}.")
    print(f"Fetched {len(neuron_df)} neurons from neuPrint.")
    return neuron_df


def skeleton_tree_coords(skel_df: pd.DataFrame) -> tuple[list, list, list]:
    """Convert an SWC table into Plotly line coordinates using parent-child edges.

    Roots (``link`` not present as a ``rowId``) are omitted so branches are
    not chained in table order.
    """
    required = {"rowId", "link", "x", "y", "z"}
    missing = required - set(skel_df.columns)
    if missing:
        raise ValueError(f"Skeleton table missing columns: {sorted(missing)}")

    nodes = skel_df[["rowId", "x", "y", "z"]].drop_duplicates("rowId")
    child = skel_df[["rowId", "link", "x", "y", "z"]].rename(
        columns={"x": "x_c", "y": "y_c", "z": "z_c"}
    )
    parent = nodes.rename(columns={"x": "x_p", "y": "y_p", "z": "z_p", "rowId": "parentId"})
    merged = child.merge(parent, left_on="link", right_on="parentId", how="inner")
    if merged.empty:
        return [], [], []

    xs: list = []
    ys: list = []
    zs: list = []
    for xc, yc, zc, xp, yp, zp in merged[["x_c", "y_c", "z_c", "x_p", "y_p", "z_p"]].itertuples(
        index=False, name=None
    ):
        xs.extend([xc, xp, None])
        ys.extend([yc, yp, None])
        zs.extend([zc, zp, None])
    return xs, ys, zs


def type_color_map(types: list[str]) -> dict[str, str]:
    """Map neuron types to a qualitative palette (same type → same color)."""
    palette = list(qualitative.Dark24) + list(qualitative.Light24) + list(qualitative.Alphabet)
    unique = sorted({t for t in types})
    if not unique:
        return {}
    return {name: palette[i % len(palette)] for i, name in enumerate(unique)}


def neuron_type(row) -> str:
    value = row.get("type") if isinstance(row, dict) else row["type"] if "type" in row else None
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def parse_obj_mesh(obj_bytes: bytes) -> tuple[list, list, list, list, list, list]:
    """Parse an OBJ mesh into Plotly ``Mesh3d`` vertex/face arrays."""
    text = obj_bytes.decode("utf-8", errors="replace") if isinstance(obj_bytes, (bytes, bytearray)) else str(obj_bytes)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
            zs.append(float(parts[3]))
        elif line.startswith("f "):
            verts = []
            for token in line.split()[1:]:
                idx = int(token.split("/")[0])
                if idx < 0:
                    idx = len(xs) + idx + 1
                verts.append(idx - 1)
            if len(verts) < 3:
                continue
            for t in range(1, len(verts) - 1):
                i.append(verts[0])
                j.append(verts[t])
                k.append(verts[t + 1])
    return xs, ys, zs, i, j, k


def _synapse_counts_from_roi_info(raw) -> dict[str, int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            raw = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for roi, payload in raw.items():
        if isinstance(payload, dict):
            counts[str(roi)] = int(payload.get("pre", 0) or 0) + int(payload.get("post", 0) or 0)
        else:
            try:
                counts[str(roi)] = int(payload)
            except (TypeError, ValueError):
                continue
    return counts


def most_synaptic_roi(neurons: pd.DataFrame) -> str | None:
    """Pick the ROI with the most synapses across the loaded subset."""
    totals: dict[str, int] = defaultdict(int)
    if "roiInfo" in neurons.columns:
        for raw in neurons["roiInfo"]:
            for roi, count in _synapse_counts_from_roi_info(raw).items():
                totals[roi] += count

    reserved = {
        "bodyId",
        "type",
        "instance",
        "status",
        "statusLabel",
        "cropped",
        "roiInfo",
        "somaLocation",
        "inputRois",
        "outputRois",
    }
    for col in neurons.columns:
        if col in reserved:
            continue
        series = pd.to_numeric(neurons[col], errors="coerce")
        if series.notna().any() and series.dtype.kind in "iuf":
            totals[str(col)] += int(series.fillna(0).sum())

    if not totals:
        return None
    return max(totals, key=totals.get)


def choose_outline_roi(client, neurons: pd.DataFrame) -> str | None:
    """Prefer a whole-brain/CNS ROI; else the busiest ROI in this subset."""
    available = set()
    try:
        from neuprint import fetch_all_rois

        available = set(fetch_all_rois(client=client) or [])
    except Exception as exc:  # pragma: no cover - network/server specific
        warnings.warn(f"Could not list ROIs: {exc}")

    for name in OUTLINE_ROI_CANDIDATES:
        if name in available:
            # Whole-CNS/brain outline, if neuPrint lists it as a single ROI.
            print(f"Using outline ROI {name!r} (whole-CNS/brain candidate present in dataset).")
            return name

    fallback = most_synaptic_roi(neurons)
    if fallback:
        # Fallback: no single CNS outline ROI; use the neuropil with the
        # most synapses in the loaded subset so the hull still frames the cells.
        print(
            f"No whole-CNS outline ROI found; using {fallback!r} "
            "(ROI with the most synapses in the loaded subset)."
        )
        return fallback

    warnings.warn("Could not determine an ROI for the brain hull.")
    return None


def fetch_outline_mesh(client, roi: str):
    try:
        data = client.fetch_roi_mesh(roi)
    except Exception as exc:
        warnings.warn(f"fetch_roi_mesh({roi!r}) failed: {exc}")
        return None
    if not data:
        warnings.warn(f"fetch_roi_mesh({roi!r}) returned no data.")
        return None
    return data


def collect_skeleton_traces(neurons: pd.DataFrame, client) -> tuple[list[go.Scatter3d], int]:
    colors = type_color_map([neuron_type(row) for _, row in neurons.iterrows()])
    traces: list[go.Scatter3d] = []
    skipped = 0
    seen_types: set[str] = set()

    for _, row in neurons.iterrows():
        body_id = int(row["bodyId"])
        ntype = neuron_type(row)
        try:
            skel = fetch_skeletons(body_id, client=client)
        except Exception as exc:
            warnings.warn(f"No skeleton for bodyId {body_id}: {exc}")
            skipped += 1
            continue
        if skel is None or getattr(skel, "empty", False) or len(skel) == 0:
            warnings.warn(f"Empty skeleton for bodyId {body_id}; skipping.")
            skipped += 1
            continue

        xs, ys, zs = skeleton_tree_coords(skel)
        if not xs:
            warnings.warn(f"Skeleton for bodyId {body_id} has no parent-child edges; skipping.")
            skipped += 1
            continue

        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=ntype,
                legendgroup=ntype,
                showlegend=ntype not in seen_types,
                line=dict(color=colors[ntype], width=3),
                hovertemplate=f"type={ntype}<br>bodyId={body_id}<extra></extra>",
            )
        )
        seen_types.add(ntype)

    return traces, skipped


def mesh_trace(obj_bytes: bytes, roi: str) -> go.Mesh3d | None:
    xs, ys, zs, i, j, k = parse_obj_mesh(obj_bytes)
    if not xs or not i:
        warnings.warn(f"ROI mesh {roi!r} could not be parsed as OBJ faces.")
        return None
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        color=MESH_COLOR,
        opacity=MESH_OPACITY,
        name=f"ROI {roi}",
        hoverinfo="skip",
        showlegend=True,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.3, specular=0.0),
    )


def build_figure(traces: list, title: str) -> go.Figure:
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        legend_title_text="type",
        scene=dict(
            aspectmode="data",
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D MaleCNS connectome skeleton viewer.")
    parser.add_argument(
        "--with-mesh",
        action="store_true",
        help="Overlay a transparent brain/CNS ROI mesh under the skeletons.",
    )
    parser.add_argument(
        "--input",
        default="connectome_export/neurons.csv",
        help="Path to neurons.csv (default: connectome_export/neurons.csv).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write an HTML file instead of opening the browser.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    neurons = load_neurons(Path(args.input))
    client = get_client(SERVER, DATASET)

    traces, skipped = collect_skeleton_traces(neurons, client)
    if not traces:
        raise SystemExit("No skeletons could be loaded; nothing to plot.")

    title = f"MaleCNS {DATASET} skeletons ({TYPE_REGEX})"
    if args.with_mesh:
        roi = choose_outline_roi(client, neurons)
        if roi:
            mesh_bytes = fetch_outline_mesh(client, roi)
            if mesh_bytes:
                mesh = mesh_trace(mesh_bytes, roi)
                if mesh is not None:
                    traces = [mesh, *traces]
                    title += f" + ROI mesh {roi}"
            else:
                warnings.warn("Continuing without a brain hull (mode 1).")
        else:
            warnings.warn("Continuing without a brain hull (mode 1).")

    fig = build_figure(traces, title)
    print(f"Plotted {len(traces)} traces; skipped {skipped} bodyIds without usable skeletons.")

    if args.output:
        out = Path(args.output)
        fig.write_html(str(out), include_plotlyjs=True, full_html=True)
        print(f"Wrote {out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
