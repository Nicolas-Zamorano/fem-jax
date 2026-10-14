"""
Reference configurations for two patch families whose raw weights already form a
partition of unity (S = sum_i phi_i = 1), so that the raw span contains the constants.

  Option 1  criss-cross family : vertex stars of the criss-cross triangulation, fan weights
            (interior node = diamond, interior cell = square, mid-side = half-diamond,
             corner = quarter-diamond)
  Option 2  bilinear family    : the square patches of the l-infinity example, with the
            tensor-product (Q1) hat as weight (interior, mid-side, corner)

A third figure assembles each family on (0,1)^2 and plots S, next to the original
l-infinity cones, for which S oscillates between 1 and 2.

Lengths are in units of the grid spacing s (s = 1). Every configuration is drawn in
local coordinates with its centre at the origin, at the same scale.

Usage:  python plot_configurations.py [--outdir figures] [--n 3] [--show]
Output: option1_crisscross.{pdf,png}, option2_bilinear.{pdf,png}, family_sums.{pdf,png}
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

# ----------------------------------------------------------------------------- style
INK = "#52514E"  # text / thin lines (talk's \inkc)
DARK = "#1f1e1c"  # Dirichlet side
CENTRE = "#EB6834"  # centre marker (talk's \rawcol)
# single-hue sequential ramp, white = 0 (recedes to the page), dark blue = 1
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "weight", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
LEVELS = np.linspace(0.0, 1.0, 11)  # 10 bands: band edges are the level sets

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "cm",
    }
)


# ----------------------------------------------------------------------------- data
Segment = tuple[tuple[float, float], tuple[float, float]]


@dataclass
class Config:
    """A reference configuration: a convex patch, its weight, and what to draw on it."""

    title: str
    formula: str  # mathtext label of the weight
    polygon: list[tuple[float, float]]  # patch vertices, counter-clockwise
    weight: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dirichlet: list[Segment] = field(default_factory=list)  # Dirichlet side(s)
    edges: list[Segment] = field(default_factory=list)  # fan edges / kink lines
    edge_label: str = "fan edges"
    centre: tuple[float, float] = (0.0, 0.0)


def criss_cross_configs() -> list[Config]:
    """Option 1: vertex stars of the criss-cross mesh (each cell cut by both diagonals)."""
    h = 0.5
    diamond = [(1, 0), (h, h), (0, 1), (-h, h), (-1, 0), (-h, -h), (0, -1), (h, -h)]
    square = [(h, h), (-h, h), (-h, -h), (h, -h)]
    half_diamond = [(-1, 0), (1, 0), (h, h), (0, 1), (-h, h)]
    quarter_diamond = [(0, 0), (1, 0), (h, h), (0, 1)]
    fan = lambda verts: [((0.0, 0.0), v) for v in verts if v != (0, 0)]
    return [
        Config(
            "Interior node: diamond",
            r"$\varphi = 1-\|x-c\|_1/s$",
            diamond,
            lambda x, y: 1 - np.abs(x) - np.abs(y),
            edges=fan(diamond),
        ),
        Config(
            "Interior cell: square",
            r"$\varphi = 1-2\|x-c\|_\infty/s$",
            square,
            lambda x, y: 1 - 2 * np.maximum(np.abs(x), np.abs(y)),
            edges=fan(square),
        ),
        Config(
            "Mid-side: half-diamond",
            r"$\varphi = 1-\|x-c\|_1/s$",
            half_diamond,
            lambda x, y: 1 - np.abs(x) - y,
            dirichlet=[((-1, 0), (1, 0))],
            edges=fan([(h, h), (0, 1), (-h, h)]),
        ),
        Config(
            "Corner: quarter-diamond",
            r"$\varphi = 1-\|x-c\|_1/s$",
            quarter_diamond,
            lambda x, y: 1 - x - y,
            dirichlet=[((0, 0), (1, 0)), ((0, 0), (0, 1))],
            edges=fan([(h, h)]),
        ),
    ]


def bilinear_configs() -> list[Config]:
    """Option 2: the square patches of the l-infinity example with the Q1 hat."""
    hat = lambda t: np.clip(1 - np.abs(t), 0, None)
    return [
        Config(
            "Interior: square",
            r"$\varphi = (1-|x_1|)(1-|x_2|)$",
            [(1, 1), (-1, 1), (-1, -1), (1, -1)],
            lambda x, y: hat(x) * hat(y),
            edges=[((-1, 0), (1, 0)), ((0, -1), (0, 1))],
            edge_label="kink lines",
        ),
        Config(
            "Mid-side: rectangle",
            r"$\varphi = (1-|x_1|)(1-x_2)$",
            [(-1, 0), (1, 0), (1, 1), (-1, 1)],
            lambda x, y: hat(x) * hat(y),
            dirichlet=[((-1, 0), (1, 0))],
            edges=[((0, 0), (0, 1))],
            edge_label="kink lines",
        ),
        Config(
            "Corner: square",
            r"$\varphi = (1-x_1)(1-x_2)$",
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            lambda x, y: hat(x) * hat(y),
            dirichlet=[((0, 0), (1, 0)), ((0, 0), (0, 1))],
            edge_label="kink lines",
        ),
    ]


# ----------------------------------------------------------------------------- geometry
def triangulate(cfg: Config, h: float = 0.02) -> mtri.Triangulation:
    """Fine triangulation of the (convex) patch: interior grid + dense samples on the
    boundary and on the fan edges / kink lines, so the creases are resolved exactly."""
    poly = np.array(cfg.polygon, float)
    lo, hi = poly.min(0), poly.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + h, h), np.arange(lo[1], hi[1] + h, h))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = MplPath(poly).contains_points(grid, radius=-1e-9)  # strict interior

    def sample(a, b, step=h / 2):
        a, b = np.asarray(a, float), np.asarray(b, float)
        k = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        return a + np.linspace(0, 1, k)[:, None] * (b - a)

    ring = list(zip(cfg.polygon, cfg.polygon[1:] + cfg.polygon[:1]))
    extra = [sample(a, b) for a, b in ring + cfg.edges]
    pts = np.unique(np.round(np.vstack([grid[inside], *extra]), 10), axis=0)
    return mtri.Triangulation(pts[:, 0], pts[:, 1])


def outward_normal(seg: Segment, poly: list[tuple[float, float]]) -> np.ndarray:
    a, b = np.asarray(seg[0], float), np.asarray(seg[1], float)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([t[1], -t[0]])
    centroid = np.mean(np.asarray(poly, float), axis=0)
    return n if np.dot((a + b) / 2 - centroid, n) > 0 else -n


def draw_dirichlet(ax, seg: Segment, poly, tick: float = 0.07, spacing: float = 0.1):
    """Thick line plus short hatch ticks on the outer side (clamped-boundary symbol)."""
    a, b = np.asarray(seg[0], float), np.asarray(seg[1], float)
    ax.plot(
        *np.column_stack([a, b]), color=DARK, lw=2.6, solid_capstyle="butt", zorder=6
    )
    t = (b - a) / np.linalg.norm(b - a)
    n = outward_normal(seg, poly)
    d = tick * (n - t) / np.sqrt(2)
    for p in a + np.arange(spacing / 2, np.linalg.norm(b - a), spacing)[:, None] * t:
        ax.plot([p[0], p[0] + d[0]], [p[1], p[1] + d[1]], color=DARK, lw=0.8, zorder=6)


# ----------------------------------------------------------------------------- drawing
def draw_2d(ax, cfg: Config, lim: float = 1.2):
    tri = triangulate(cfg)
    z = np.clip(cfg.weight(tri.x, tri.y), 0, 1)
    cs = ax.tricontourf(
        tri, z, levels=LEVELS, cmap=WEIGHT_CMAP, vmin=0, vmax=1, zorder=1
    )

    # patch boundary: dotted where the weight vanishes, Dirichlet side drawn on top
    ring = np.array(cfg.polygon + cfg.polygon[:1], float)
    ax.plot(ring[:, 0], ring[:, 1], ls=(0, (1.2, 1.6)), color=INK, lw=1.3, zorder=4)
    for a, b in cfg.edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.8, alpha=0.85, zorder=3)
    for seg in cfg.dirichlet:
        draw_dirichlet(ax, seg, cfg.polygon)
    ax.plot(*cfg.centre, "o", ms=7, mfc=CENTRE, mec="white", mew=1.2, zorder=7)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(length=2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{cfg.title}\n{cfg.formula}", pad=6)
    return cs


def draw_3d(ax, cfg: Config, lim: float = 1.2):
    tri = triangulate(cfg, h=0.03)
    z = np.clip(cfg.weight(tri.x, tri.y), 0, 1)
    ax.computed_zorder = False  # draw in call order: surface first, then the lines
    ax.plot_trisurf(
        tri,
        z,
        cmap=WEIGHT_CMAP,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.95,
        zorder=1,
    )
    ring = np.array(cfg.polygon + cfg.polygon[:1], float)
    ax.plot(ring[:, 0], ring[:, 1], 0 * ring[:, 0], ls=":", color=INK, lw=1.0)
    for a, b in cfg.edges:  # creases, lifted onto the surface
        p = np.linspace(a, b, 40)
        ax.plot(p[:, 0], p[:, 1], cfg.weight(p[:, 0], p[:, 1]), color=INK, lw=0.7)
    for a, b in cfg.dirichlet:  # trace of phi on the Dirichlet side
        p = np.linspace(a, b, 80)
        ax.plot(p[:, 0], p[:, 1], 0 * p[:, 0], color=DARK, lw=2.2)
        ax.plot(
            p[:, 0], p[:, 1], cfg.weight(p[:, 0], p[:, 1]), color=DARK, lw=1.0, ls="--"
        )
    ax.plot([cfg.centre[0]], [cfg.centre[1]], [1.0], "o", ms=5, mfc=CENTRE, mec="white")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(
        elev=26, azim=35
    )  # look from the open side, so the Dirichlet side is at the back
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([0, 1])
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d9d8d4")
        axis._axinfo["grid"]["color"] = "#ecebe8"


def legend_handles(configs: list[Config]):
    labels = sorted({c.edge_label for c in configs})
    return [
        Line2D(
            [],
            [],
            color=INK,
            ls=(0, (1.2, 1.6)),
            lw=1.3,
            label=r"patch boundary ($\varphi=0$)",
        ),
        Line2D([], [], color=DARK, lw=2.6, label=r"Dirichlet side $\hat\Gamma$"),
        *[Line2D([], [], color=INK, lw=0.8, label=l) for l in labels],
        Line2D(
            [],
            [],
            color=DARK,
            lw=1.0,
            ls="--",
            label=r"$\varphi$ along $\hat\Gamma$ (3D)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=7,
            mfc=CENTRE,
            mec="white",
            label=r"centre $c$",
        ),
    ]


def plot_configs(configs: list[Config], title: str, path: Path):
    k = len(configs)
    fig = plt.figure(figsize=(3.0 * k + 0.9, 6.6))
    gs = fig.add_gridspec(
        2,
        k + 1,
        width_ratios=[1] * k + [0.06],
        height_ratios=[1.0, 0.9],
        hspace=0.12,
        wspace=0.18,
    )
    cs = None
    for j, cfg in enumerate(configs):
        cs = draw_2d(fig.add_subplot(gs[0, j]), cfg)
        draw_3d(fig.add_subplot(gs[1, j], projection="3d"), cfg)
    cax = fig.add_subplot(gs[0, k])
    cb = fig.colorbar(cs, cax=cax, ticks=[0, 0.5, 1])
    cb.set_label(r"weight $\varphi$")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.legend(
        handles=legend_handles(configs),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.04, right=0.92, top=0.86, bottom=0.07)
    save(fig, path)


# ----------------------------------------------------------------------------- families
def family_sum(kind: str, n: int, X: np.ndarray, Y: np.ndarray):
    """S = sum of the raw weights of a whole family on (0,1)^2, and the patch centres."""
    s = 1.0 / n
    nodes = [(i * s, j * s) for i in range(n + 1) for j in range(n + 1)]
    S = np.zeros_like(X)
    if kind == "linf":
        for cx, cy in nodes:
            S += np.clip(1 - np.maximum(abs(X - cx), abs(Y - cy)) / s, 0, None)
        return S, nodes, []
    if kind == "bilinear":
        for cx, cy in nodes:
            S += np.clip(1 - abs(X - cx) / s, 0, None) * np.clip(
                1 - abs(Y - cy) / s, 0, None
            )
        return S, nodes, []
    if kind == "crisscross":
        cells = [((i + 0.5) * s, (j + 0.5) * s) for i in range(n) for j in range(n)]
        for cx, cy in nodes:
            S += np.clip(1 - (abs(X - cx) + abs(Y - cy)) / s, 0, None)
        for cx, cy in cells:
            S += np.clip(1 - 2 * np.maximum(abs(X - cx), abs(Y - cy)) / s, 0, None)
        return S, nodes, cells
    raise ValueError(kind)


def plot_family_sums(n: int, path: Path):
    t = np.linspace(0, 1, 601)
    X, Y = np.meshgrid(t, t)
    panels = [
        ("linf", r"Original: $\ell^\infty$ cones"),
        ("crisscross", "Option 1: criss-cross"),
        ("bilinear", "Option 2: bilinear hats"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "S", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    fig = plt.figure(figsize=(11.0, 4.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15)
    im = None
    for j, (kind, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        S, nodes, cells = family_sum(kind, n, X, Y)
        im = ax.imshow(
            S,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            vmin=1,
            vmax=2,
            interpolation="bilinear",
        )
        for k in range(n + 1):  # grid lines
            ax.plot([k / n, k / n], [0, 1], color="white", lw=0.6, alpha=0.7)
            ax.plot([0, 1], [k / n, k / n], color="white", lw=0.6, alpha=0.7)
        ax.plot(
            *np.array(nodes).T,
            "o",
            ms=5,
            mfc=CENTRE,
            mec="white",
            mew=0.9,
            clip_on=False,
            label="node centres",
        )
        if cells:
            ax.plot(
                *np.array(cells).T,
                "s",
                ms=4.5,
                mfc="white",
                mec=CENTRE,
                mew=1.2,
                label="cell centres",
            )
        ax.set_title(f"{title}\n$S\\in[{S.min():.2f},\\,{S.max():.2f}]$", pad=6)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(length=2, labelsize=8)
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_color(DARK)
            sp.set_linewidth(1.6)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), ticks=[1, 1.5, 2])
    cb.ax.set_title(r"$S$", fontsize=11, pad=8)
    cb.outline.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=5,
            mfc=CENTRE,
            mec="white",
            label="node centres",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="s",
            ms=4.5,
            mfc="white",
            mec=CENTRE,
            mew=1.2,
            label="cell centres (option 1)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle(
        f"Sum of the raw weights on $(0,1)^2$, $n={n}$",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.04, right=0.94, top=0.80, bottom=0.12)
    save(fig, path)


# ----------------------------------------------------------------------------- main
def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    print("wrote", path.with_suffix(".pdf"), "and .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default="figures")
    ap.add_argument(
        "--n", type=int, default=1, help="grid size for the family-sum figure"
    )
    ap.add_argument("--show", action="store_true", help="also open the figures")
    args = ap.parse_args()
    out = Path(args.outdir)

    # sanity check: each option is a partition of unity (S = 1 on the closed square)
    t = np.linspace(0, 1, 401)
    X, Y = np.meshgrid(t, t)
    for kind in ("crisscross", "bilinear"):
        S, *_ = family_sum(kind, args.n, X, Y)
        assert np.allclose(S, 1.0), f"{kind}: S is not identically 1"

    plot_configs(
        criss_cross_configs(),
        "Option 1: criss-cross family (fan weights, $S\\equiv1$)",
        out / "option1_crisscross",
    )
    plot_configs(
        bilinear_configs(),
        "Option 2: bilinear hats on the square patches ($S\\equiv1$)",
        out / "option2_bilinear",
    )
    plot_family_sums(args.n, out / "family_sums")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
