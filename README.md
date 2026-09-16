# flybrain3D

Interactive 3D viewer for a MaleCNS fly-brain connectome subset from
[neuPrint](https://neuprint.janelia.org) (`male-cns:v1.0`).

## Setup

```bash
pip install -r requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS='<your neuPrint token>'
```

Create a token in the neuPrint UI (account menu).

## Export a neuron subset

```bash
python fetch_connectome.py
```

Default subset: descending neurons matching `DNa.*` (capped at 50 cells).
Writes `connectome_export/neurons.csv`, `connections.csv`, and `graph.json`.

## View skeletons

```bash
python viewer.py
python viewer.py --with-mesh
python viewer.py --with-mesh --output viewer.html
```

- Mode 1 draws SWC skeletons as 3D polylines along the parent–child tree, colored by `type`.
- Mode 2 adds a transparent ROI hull (whole CNS/brain if neuPrint has that mesh; otherwise the ROI with the most synapses in the loaded subset).
- Neurons are read from `connectome_export/neurons.csv` when that file exists, so the viewer does not re-query the type regex.
