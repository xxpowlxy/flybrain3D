"""Load a MaleCNS neuron subset from neuPrint and export it to disk.

Default subset: descending neurons matching ``DNa.*`` on
``neuprint.janelia.org`` / ``male-cns:v1.0``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

SERVER = "neuprint.janelia.org"
DATASET = "male-cns:v1.0"
TYPE_REGEX = "DNa.*"
MAX_NEURONS = 50
EXPORT_DIR = Path("connectome_export")


def get_client(server: str = SERVER, dataset: str = DATASET):
    """Return a neuPrint client using ``NEUPRINT_APPLICATION_CREDENTIALS``."""
    from neuprint import Client

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    return Client(server, dataset=dataset, token=token)


def fetch_neurons(type_regex: str = TYPE_REGEX, max_neurons: int | None = MAX_NEURONS, client=None):
    """Fetch neurons matching ``type_regex``, optionally truncated to ``max_neurons``.

    This wraps ``neuprint.fetch_neurons`` so ``viewer.py`` can reuse the same
    subset pattern without duplicating query logic.
    """
    from neuprint import NeuronCriteria
    from neuprint import fetch_neurons as neuprint_fetch_neurons

    if client is None:
        client = get_client()

    neuron_df, roi_counts_df = neuprint_fetch_neurons(
        NeuronCriteria(type=type_regex),
        client=client,
    )
    if neuron_df is None or neuron_df.empty:
        return neuron_df, roi_counts_df

    neuron_df = neuron_df.sort_values("bodyId").reset_index(drop=True)
    if max_neurons is not None and len(neuron_df) > max_neurons:
        neuron_df = neuron_df.head(int(max_neurons)).reset_index(drop=True)
        if roi_counts_df is not None and not roi_counts_df.empty and "bodyId" in roi_counts_df.columns:
            keep = set(neuron_df["bodyId"].tolist())
            roi_counts_df = roi_counts_df[roi_counts_df["bodyId"].isin(keep)].reset_index(drop=True)
    return neuron_df, roi_counts_df


def fetch_connections(body_ids, client=None) -> pd.DataFrame:
    """Fetch directed synapses among ``body_ids`` (empty if the set is too small)."""
    from neuprint import NeuronCriteria, fetch_adjacencies

    ids = [int(x) for x in body_ids]
    if len(ids) < 2:
        return pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "roi", "weight"])

    if client is None:
        client = get_client()

    _neurons, conn_df = fetch_adjacencies(
        NeuronCriteria(bodyId=ids),
        NeuronCriteria(bodyId=ids),
        client=client,
    )
    if conn_df is None or conn_df.empty:
        return pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "roi", "weight"])
    return conn_df


def _graph_payload(neuron_df: pd.DataFrame, conn_df: pd.DataFrame) -> dict:
    nodes = []
    for row in neuron_df.itertuples(index=False):
        nodes.append(
            {
                "bodyId": int(row.bodyId),
                "type": None if pd.isna(getattr(row, "type", None)) else str(row.type),
                "instance": None
                if not hasattr(row, "instance") or pd.isna(row.instance)
                else str(row.instance),
            }
        )

    edges = []
    if conn_df is not None and not conn_df.empty:
        grouped = (
            conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
            if "weight" in conn_df.columns
            else conn_df
        )
        for row in grouped.itertuples(index=False):
            edges.append(
                {
                    "source": int(row.bodyId_pre),
                    "target": int(row.bodyId_post),
                    "weight": int(getattr(row, "weight", 0) or 0),
                }
            )
    return {"nodes": nodes, "edges": edges}


def export_connectome(
    export_dir: Path = EXPORT_DIR,
    type_regex: str = TYPE_REGEX,
    max_neurons: int | None = MAX_NEURONS,
    client=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query neuPrint and write ``neurons.csv``, ``connections.csv``, ``graph.json``."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = get_client()

    neuron_df, _roi_counts = fetch_neurons(type_regex=type_regex, max_neurons=max_neurons, client=client)
    if neuron_df is None or neuron_df.empty:
        raise SystemExit(f"No neurons matched type regex {type_regex!r} in {DATASET}.")

    conn_df = fetch_connections(neuron_df["bodyId"], client=client)

    neuron_path = export_dir / "neurons.csv"
    conn_path = export_dir / "connections.csv"
    graph_path = export_dir / "graph.json"

    neuron_df.to_csv(neuron_path, index=False)
    conn_df.to_csv(conn_path, index=False)
    graph_path.write_text(json.dumps(_graph_payload(neuron_df, conn_df), indent=2), encoding="utf-8")

    print(f"Wrote {len(neuron_df)} neurons -> {neuron_path}")
    print(f"Wrote {len(conn_df)} connection rows -> {conn_path}")
    print(f"Wrote graph -> {graph_path}")
    return neuron_df, conn_df


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a MaleCNS neuron subset from neuPrint.")
    parser.add_argument("--type-regex", default=TYPE_REGEX, help="Neuron type regex (default: DNa.*).")
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=MAX_NEURONS,
        help="Maximum number of neurons to export (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_DIR),
        help="Directory for neurons.csv / connections.csv / graph.json.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    export_connectome(
        export_dir=Path(args.output_dir),
        type_regex=args.type_regex,
        max_neurons=args.max_neurons,
    )


if __name__ == "__main__":
    main()
