# Interactive results site

This folder is a self-contained, static GitHub Pages site that lets readers
explore the phonological-competition results interactively (Plotly, no backend).

**Live:** https://vocaliodmiku.github.io/HTP-benchmark/

## Contents

| File | Purpose |
|------|---------|
| `index.html` | Page shell + four views (Overview, Trajectory Explorer, Model Grid, Metrics) |
| `app.js` | Loads the JSON in `data/` and renders the charts/table |
| `style.css` | Styling |
| `build_data.py` | Regenerates `data/*.json` from the project's result CSVs |
| `data/trajectories.json` | Per-model activation curves + human reference (for plotting) |
| `data/metrics.json` | RMSE / MAE / correlation per model per competitor type |
| `data/human.json` | Human VWP reference curves on their own |
| `.nojekyll` | Tells GitHub Pages to serve the folder verbatim |

## Rebuilding the data

After re-running experiments (i.e. new `competition_mean.csv` files), regenerate
the JSON from the repo root:

```bash
python docs/build_data.py
```

The script reads each model's `experiments/.../competition_mean.csv` and the human
reference (`notebooks/INPUT/amt_human_mean.csv`). The RMSE/MAE pipeline is a port of
`notebooks/calculate_RMSE_MAE.ipynb`, so the site's numbers match the paper's notebook.
To add or remove a model, edit the `MODELS` registry at the top of `build_data.py`.

## Previewing locally

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000
```

(Serve over HTTP — opening `index.html` with `file://` blocks the `fetch()` calls.)

## Enabling GitHub Pages

Repo **Settings → Pages → Build and deployment → Deploy from a branch**, then choose
branch `main` and folder `/docs`. The site publishes at the URL above within a minute.
