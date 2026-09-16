# flybrain3D

3D viewer for a fly-brain connectome subset from
[neuPrint](https://neuprint.janelia.org) (`male-cns:v1.0`).

Copy `.env.example` to `.env` and paste your neuPrint token:

```
NEUPRINT_APPLICATION_CREDENTIALS=<your neuPrint token>
```

```bash
uv run python fetch_connectome.py
uv run python viewer.py
uv run python viewer.py --with-mesh
```

`fetch_connectome.py` writes neurons to `connectome_export/`. The viewer uses that folder if it exists.
