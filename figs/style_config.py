"""
Shared style configuration for all manuscript figures.

Usage in notebooks:
    import sys
    sys.path.insert(0, str(Path('..').resolve()))
    import style_config as sc

    ax.plot(x, y, **sc.STYLES["saddle"])
    ax.axvline(x, **sc.STYLES["F_max"])
    ax.legend(**sc.LEGEND_KW, loc="lower left")
"""
from pathlib import Path
import matplotlib.pyplot as plt

# Apply shared mplstyle on import
_STYLE_PATH = Path(__file__).parent / "paper.mplstyle"
if _STYLE_PATH.exists():
    plt.style.use(str(_STYLE_PATH))

# ── Page geometry ──────────────────────────────────────────────
TEXT_WIDTH = 6.5  # inches

# ── Standard figure sizes ──────────────────────────────────────
FIGSIZE_2x2 = (TEXT_WIDTH, 4.5)
FIGSIZE_SINGLE = (0.6 * TEXT_WIDTH, 0.45 * TEXT_WIDTH)

# ── Font sizes ─────────────────────────────────────────────────
PANEL_TITLE_SIZE = 8
AXIS_LABEL_SIZE = 7
TICK_LABEL_SIZE = 7
LEGEND_FONTSIZE = 6
ANNOTATION_FONTSIZE = 7

# Apply font sizes via rcParams so they take effect automatically
plt.rcParams["axes.titlesize"] = PANEL_TITLE_SIZE
plt.rcParams["axes.labelsize"] = AXIS_LABEL_SIZE
plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE

# ── Legend defaults ────────────────────────────────────────────
LEGEND_KW = dict(fontsize=LEGEND_FONTSIZE, handlelength=2)

# ── Sigmoid palette (fig5) ────────────────────────────────────
COLOR_SIGMOID = ["C4", "C5", "C6"]

# ── Hexbin / colorbar (fig6) ──────────────────────────────────
HEXBIN_KW = dict(gridsize=60, cmap="viridis", linewidths=0.1,
                 edgecolors="face")
COLORBAR_KW = dict(shrink=0.85, pad=0.02)

# ── Plot styles ────────────────────────────────────────────────
# Each entry is a dict of kwargs suitable for ax.plot() or ax.axvline().
STYLES = {
    "saddle": dict(color="C1", ls="-",  lw=2, label="saddle", zorder=0),
    "bulk":   dict(color="C8", ls="--", lw=1, label="bulk",   zorder=1),
    "peak":   dict(color="C0", ls="--", lw=1, label="peak",   zorder=2),
    "F_mean": dict(color="C2", ls=":",  lw=1, label=r"$F_\mathrm{mean}$", zorder=-4),
    "F_max":  dict(color="C3", ls=":",  lw=1, label=r"$F_\mathrm{max}$",  zorder=-4),
    "F_cross":dict(color="C6", ls=":",  lw=1, label=r"$F_\mathrm{cross}$",zorder=-2),
    "true":   dict(color="k",  ls="-",  lw=0.5, label="true",   zorder=10),
    "bounds": dict(color="0.5",ls="--", lw=1,                  zorder=10),
    "boundary":dict(color="C3",ls="--", lw=3),
    "phi_half":dict(color="C7", ls=":", lw=1, label=r"$\phi_\mathrm{half}$"),
}
