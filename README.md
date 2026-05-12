# 26_stubby

Public code and compiled PDFs for "Mount Fuji's stubby peak: the genotypic
density of additive landscapes near maximal fitness."

This repository is intended for reproducing the paper figures and reading the
current public manuscript PDFs. It does not include the LaTeX source,
bibliography, reviewer response materials, or other private manuscript files.

## Repository Structure

- `src/` - flat Python modules for additive landscape computations,
  saddle-point approximations, and Touzet tail counting
- `figs/` - Jupyter notebooks, generated figure PDFs/PNGs, shared styling, and
  figure-specific input data
- `manuscript/` - compiled public PDFs:
  `Kinney2026_stubby_main.pdf` and `Kinney2026_stubby_si.pdf`

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --group dev
```

## Generating Figures

Run the notebooks in this order:

```bash
uv run jupyter nbconvert --to notebook --execute figs/fig1/fig1.ipynb
uv run jupyter nbconvert --to notebook --execute figs/fig2/fig2.ipynb
uv run jupyter nbconvert --to notebook --execute figs/fig3_figS1/fig3_figS1.ipynb
uv run jupyter nbconvert --to notebook --execute figs/fig4_figS2/fig4_figS2.ipynb
uv run jupyter nbconvert --to notebook --execute figs/fig5/fig5.ipynb
uv run jupyter nbconvert --to notebook --execute figs/figS3/figS3.ipynb
```

`fig2` should be run before `fig4_figS2` and `fig5`, because it writes the
shared simulated `theta` arrays used by those notebooks. Touzet `.npz` cache
files may be generated locally during execution; they are ignored by git.
