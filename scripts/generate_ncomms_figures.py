#!/usr/bin/env python3
"""Generate the Nature Communications main-text figures from the result JSON files.

Conventions follow the Nature Portfolio figure guidelines:
  * 180 mm double-column width, Arial/Helvetica, 5-7 pt type, bold lower-case panel labels
  * no truncated bar axes, no dual y-axes, colour-blind-safe palette (Okabe-Ito)
  * colour encodes the functional family of a role; role identity is always written out
  * every plotted number is exported by scripts/generate_source_data.py
"""

from __future__ import annotations

import json
import os
import sys
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402

MM = 1 / 25.4
FULL_WIDTH = 180 * MM

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
    }
)

# Okabe-Ito, validated for deutan/protan/tritan separation.
# Two palettes that must not be confused. SETTING_COLORS identifies the empirical setting
# (a designation programme or a public appeal) and FAMILY_COLORS identifies a learned role.
# They previously shared two hues, so orange meant "sanctions" in one panel and "hub-like
# role" in the next; the role hues are now disjoint from the setting hues. Both are drawn
# from Okabe-Ito and both pass the colour-vision check in scripts/validate_palette.py.
FAMILY_COLORS = {
    "sender": "#332288",  # send-dominant, peripheral donors
    "receiver": "#117733",  # receive-dominant participants, sinks
    "relay": "#E69F00",  # balanced low-degree intermediaries
    "hub": "#CC79A7",  # high-degree bilateral hubs and core collection
}
FAMILY_LABELS = {
    "sender": "send-dominant (donor-like)",
    "receiver": "receive-dominant (collector-like)",
    "relay": "balanced low-degree (relay-like)",
    "hub": "high-degree bilateral (hub-like)",
}
FAMILY_ORDER = ["sender", "relay", "receiver", "hub"]

INK = "#1F2937"
INK_2 = "#4B5563"
MUTED = "#9CA3AF"
GRID = "#E5E7EB"
HIGHLIGHT = "#D55E00"
NEUTRAL = "#8A8F98"
# Okabe-Ito vermillion and blue for the two settings the paper contrasts, with a dark
# neutral for the OFAC programmes. Checked for colour-vision deficiency: the smallest
# perceptual separation of any pair is 14.7 (OKLab x100) under tritanopia and 16.1 under
# normal vision, against a floor of 15 for normal vision and 8 under simulated deficiency.
SETTING_COLORS = {
    "sanctions": "#D55E00",
    "fundraising": "#0072B2",
    "ofac": "#525252",
}

DATASETS = [
    ("israel", "NBCTF Israel (TRON)", "wcfrm_israel_tron_analysis.json", "sanctions"),
    ("ukr_tron", "Ukraine aid (TRON)", "wcfrm_ukraine_tron_analysis.json", "fundraising"),
    ("ukr_eth", "Ukraine aid (ETH)", "wcfrm_ukraine_eth_analysis.json", "fundraising"),
    ("ofac_iran", "OFAC Iran (TRON)", "wcfrm_ofac_iran_tron_analysis.json", "ofac"),
    ("ofac_ru", "OFAC Russia–Ukr. (TRON)", "wcfrm_ofac_russia_ukraine_tron_analysis.json", "ofac"),
    ("ofac_tf", "OFAC terrorism (TRON)", "wcfrm_ofac_terrorist_financing_tron_analysis.json", "ofac"),
]

FAMILY_SHORT = {"sender": "send-dominant", "receiver": "receive-dominant", "relay": "relay-like", "hub": "hub-like"}


class _RoleNames(dict):
    """Role labels derived from the functional family of each role in the primary network."""

    def __missing__(self, key):
        return f"role {key}"


ROLE_NAMES_ISRAEL = _RoleNames()


# --------------------------------------------------------------------------- data helpers
def load_json(name: str):
    with find(name).open() as f:
        return json.load(f)


def role_family(avg_in: float, avg_out: float) -> str:
    """Assign a functional family from mean in/out-degree (used for colour only)."""
    if np.isnan(avg_in) or np.isnan(avg_out):
        return "relay"
    if max(avg_in, avg_out) >= 8:
        return "hub"
    if avg_out >= 1.5 * max(avg_in, 1e-9):
        return "sender"
    if avg_in >= 1.5 * max(avg_out, 1e-9):
        return "receiver"
    return "relay"


def role_rows(data: dict):
    rows = []
    for key, stat in sorted(data["role_stats"].items(), key=lambda kv: int(kv[0])):
        size = stat["size"]
        seed_count = stat.get("seed_count", stat.get("seed_in_role", 0))
        seed_density = 100.0 * seed_count / size if size else 0.0
        avg_in = stat.get("avg_in", np.nan)
        avg_out = stat.get("avg_out", np.nan)
        dis = data["dismantling"].get(f"role_{key}", {})
        rows.append(
            {
                "role": int(key),
                "size": size,
                "share": size / data["n_nodes"],
                "avg_in": avg_in,
                "avg_out": avg_out,
                "seed_count": seed_count,
                "seed_density": seed_density,
                "family": role_family(avg_in, avg_out),
                "connectivity": dis.get("connectivity", np.nan),
                "loss": 100 * (1 - dis.get("connectivity", np.nan)),
                "n_components": dis.get("n_components", np.nan),
                "path_efficiency": dis.get("path_efficiency", np.nan),
            }
        )
    return rows


def anchor_role(data: dict) -> int:
    return int(data.get("seed_cluster", max(role_rows(data), key=lambda r: r["seed_density"])["role"]))


# --------------------------------------------------------------------------- drawing helpers
def panel_label(ax, label: str, dx: float = -0.12, dy: float = 1.04):
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom", ha="left", color="black")


def light_grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def family_legend(ax, loc="lower right", ncol=1, families=FAMILY_ORDER, **kw):
    handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=FAMILY_LABELS[f]) for f in families]
    return ax.legend(handles=handles, loc=loc, ncol=ncol, handlelength=1.0, handleheight=0.8, **kw)


def rounded_box(ax, x, y, w, h, color, text, fontsize=6.2, fill_alpha="1A", lw=0.8, text_color=INK, bold_first=False):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=lw, edgecolor=color, facecolor=color + fill_alpha, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color, linespacing=1.3, zorder=3)
    return box


def arrow(ax, p0, p1, color=NEUTRAL, lw=0.9, style="-|>", scale=7, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=scale, lw=lw, color=color, zorder=4, **kw))


# --------------------------------------------------------------------------- Figure 1
def draw_toy_network(ax, x0, y0, w, h):
    """A schematic directed transfer graph with functional roles coloured."""
    rng = np.random.default_rng(3)
    # node positions in the local box (0..1)
    donors = [(0.07, 0.86), (0.05, 0.62), (0.09, 0.38), (0.06, 0.14), (0.24, 0.94), (0.22, 0.08)]
    relays = [(0.30, 0.70), (0.30, 0.30)]
    hub = (0.52, 0.50)
    exch = (0.78, 0.78)
    sink = (0.80, 0.22)
    outer = [(0.96, 0.92), (0.97, 0.62), (0.97, 0.06)]

    def P(p):
        return (x0 + p[0] * w, y0 + p[1] * h)

    def edge(a, b, lw, color=NEUTRAL):
        ax.add_patch(FancyArrowPatch(P(a), P(b), arrowstyle="-|>", mutation_scale=5, lw=lw, color=color, alpha=0.9, zorder=2, shrinkA=2.5, shrinkB=3))

    for i, d in enumerate(donors):
        target = relays[0] if d[1] > 0.5 else relays[1]
        if i in (1, 3):
            target = hub
        edge(d, target, 0.5)
    edge(relays[0], hub, 0.9)
    edge(relays[1], hub, 0.9)
    edge(hub, exch, 1.4)
    edge(exch, hub, 0.7)
    edge(hub, sink, 1.2)
    edge(exch, outer[0], 0.6)
    edge(exch, outer[1], 0.6)
    edge(sink, outer[2], 0.5)

    def node(p, r, color, ec="white"):
        ax.add_patch(Circle(P(p), r, facecolor=color, edgecolor=ec, lw=0.5, zorder=3))

    for d in donors:
        node(d, 0.011, FAMILY_COLORS["sender"])
    for r in relays:
        node(r, 0.012, FAMILY_COLORS["relay"])
    node(hub, 0.02, FAMILY_COLORS["hub"])
    node(exch, 0.017, FAMILY_COLORS["hub"])
    node(sink, 0.014, FAMILY_COLORS["receiver"])
    for o in outer:
        node(o, 0.010, FAMILY_COLORS["receiver"])
    # anchor marker on the hub (disclosed address)
    hx, hy = P(hub)
    ax.plot(hx, hy, marker="*", ms=5, color="white", mec=INK, mew=0.3, zorder=5)
    ax.text(hx, hy - 0.075, "anchor", ha="center", va="top", fontsize=6.0, color=INK_2, zorder=5)


CONTROL_TAGS = {"israel": "israel_tron", "ukr_tron": "ukraine_tron", "ukr_eth": "ukraine_eth", "ofac_iran": "ofac_iran_tron", "ofac_ru": "ofac_russia_ukraine_tron", "ofac_tf": "ofac_terrorist_financing_tron"}


def unique_edges(key: str, data: dict) -> int:
    """Unique directed address pairs, taken from dismantling_controls.json when available."""
    path = find("dismantling_controls.json", required=False)
    if path.exists():
        ctrl = json.load(open(path))
        if CONTROL_TAGS[key] in ctrl:
            return int(ctrl[CONTROL_TAGS[key]]["n_edges"])
    return int(data[key]["n_edges"])


def generate_model_performance():
    """Supplementary screening figure, computed entirely from screening_extended.json.

    Earlier versions of this figure mixed panels computed on a superseded graph build with
    panels computed on the current one; every panel here reads the same file.
    """
    ext = load_json("screening_extended.json")
    IC = "diffusion reach (ROTOR)"
    ML = "learned importance (ROTOR)"
    order = [IC, ML, "total degree", "in-degree", "out-degree", "PageRank"]
    colour = {IC: HIGHLIGHT, ML: INK, "total degree": "#0072B2",
              "in-degree": "#009E73", "out-degree": "#CC79A7", "PageRank": "#E69F00"}
    dash = {IC: "-", ML: (0, (4, 1.4, 1, 1.4)), "total degree": "-",
            "in-degree": (0, (3, 1.4)), "out-degree": (0, (1.4, 1.2)), "PageRank": (0, (5, 1.4, 1.4, 1.4))}

    fig = plt.figure(figsize=(FULL_WIDTH, 58 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.95,
                          left=0.075, right=0.985, top=0.88, bottom=0.21)
    for lab, (fx, fy) in {"a": (0.010, 0.975), "b": (0.360, 0.975), "c": (0.685, 0.975)}.items():
        fig.text(fx, fy, lab, fontsize=8.0, fontweight="bold", va="top", ha="left")
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))

    # a. recall at fixed review-set size, all addresses
    rk = ext["candidate_sets"]["all"]["methods"]
    ks = np.array([int(k) for k in rk[IC]["recall_at_k"]])
    for name in order:
        v = np.array([rk[name]["recall_at_k"][str(k)] for k in ks])
        ax_a.plot(ks, 100 * v, color=colour[name], lw=1.6 if name == IC else 1.1,
                  ls=dash[name], label=name, zorder=4)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("review-set size $K$ (addresses)")
    ax_a.set_ylabel("designated addresses recovered (%)")
    ax_a.legend(fontsize=6.0, frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.25)
    light_grid(ax_a)

    # b, c. ROC-AUC with 95% bootstrap CI, over all addresses and restricted to hop-1
    for ax, key, title in ((ax_b, "all", "all 596,633 addresses"),
                           (ax_c, "hop1", "designated vs direct counterparties")):
        m = ext["candidate_sets"][key]["methods"]
        yv = np.arange(len(order))[::-1]
        for y, name in zip(yv, order):
            r = m[name]
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=colour[name], lw=1.1, zorder=3)
            ax.plot(r["auc"], y, "o", ms=3.6, color=colour[name], mec="white", mew=0.5, zorder=4)
            ax.text(r["ci_high"] + 0.006, y, f"{r['auc']:.3f}", fontsize=6.0, va="center", color=INK_2)
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.set_yticks(yv); ax.set_yticklabels(order, fontsize=6.0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("ROC-AUC (95% bootstrap CI)")
        ax.set_title(title, fontsize=6.0, color=INK, pad=4)
        lo = min(m[nm]["ci_low"] for nm in order); hi = max(m[nm]["ci_high"] for nm in order)
        ax.set_xlim(lo - 0.03, hi + 0.06)
        light_grid(ax, axis="x")

    fig.savefig(ROOT / "fig_screening_si.pdf")
    fig.savefig(ROOT / "fig_screening_si.png", dpi=300)
    plt.close(fig)
    print("wrote fig_screening_si")

def generate_role_landscape():
    data = {key: load_json(fname) for key, _, fname, _ in DATASETS}
    israel = data["israel"]
    rows = role_rows(israel)
    for r in rows:
        ROLE_NAMES_ISRAEL[r["role"]] = FAMILY_SHORT[r["family"]]

    fig = plt.figure(figsize=(FULL_WIDTH, 128 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.46, left=0.26, right=0.86, top=0.965, bottom=0.155, height_ratios=[1.0, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])

    # ---- a. role landscape on log-log degree axes
    panel_label(ax_a, "a", dx=-0.22)
    floor = 0.04
    x = np.array([max(r["avg_out"], floor) for r in rows])
    yv = np.array([max(r["avg_in"], floor) for r in rows])
    sizes = np.array([40 + 900 * r["share"] for r in rows])
    ax_a.plot([floor, 60], [floor, 60], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_a.text(0.09, 0.09, "in = out", color=MUTED, fontsize=6.0, rotation=45, ha="center", va="bottom", rotation_mode="anchor")
    for r, xi, yi, s_ in zip(rows, x, yv, sizes):
        ax_a.scatter(xi, yi, s=s_, color=FAMILY_COLORS[r["family"]], edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
    # Several roles sit at the same mean degrees to within a pixel; label each cluster once
    # rather than stacking three labels on one marker.
    clusters: dict[tuple, list] = {}
    for r, xi, yi in zip(rows, x, yv):
        clusters.setdefault((round(np.log10(xi) / 0.18), round(np.log10(yi) / 0.18)), []).append((r, xi, yi))
    for grp in clusters.values():
        lab = ", ".join(f"R{r['role']}" for r, _, _ in sorted(grp, key=lambda g: g[0]["role"]))
        _, xi, yi = grp[0]
        ax_a.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=6.0,
                      color=INK, ha="left", va="bottom", zorder=5)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.03, 60)
    ax_a.set_ylim(0.03, 60)
    ax_a.set_xlabel("mean out-degree (log scale)")
    ax_a.set_ylabel("mean in-degree (log scale)")
    light_grid(ax_a)
    for s_, lab, yy in [(40 + 900 * 0.02, "2%", 0.30), (40 + 900 * 0.10, "10%", 0.20), (40 + 900 * 0.45, "45%", 0.055)]:
        ax_a.scatter(1.10, yy, s=s_, color="none", edgecolor=INK_2, linewidth=0.5, transform=ax_a.transAxes, clip_on=False)
        ax_a.text(1.17, yy, lab, transform=ax_a.transAxes, fontsize=6.0, va="center", ha="left", color=INK_2)
    ax_a.text(1.10, 0.38, "share of\naddresses", transform=ax_a.transAxes, fontsize=6.0, ha="center", va="bottom", color=INK_2, linespacing=1.2)
    present = [f for f in FAMILY_ORDER if any(r["family"] == f for r in rows)]
    family_legend(ax_a, loc="upper left", bbox_to_anchor=(0.0, 1.0), labelspacing=0.3, families=present)

    # ---- b. designated density vs connectivity loss per role
    # ---- c. role removal versus budget-matched random and top-degree removal
    panel_label(ax_c, "b", dx=-0.33)
    ctrl = load_json("dismantling_controls.json")["israel_tron"]["roles"]
    order = sorted(rows, key=lambda r: -r["loss"])
    yc = np.arange(len(order))[::-1]
    for yi, r in zip(yc, order):
        c = ctrl[f"role_{r['role']}"]
        rand = 100 * (1 - c["random_connectivity_mean"])
        rand_lo, rand_hi = 100 * (1 - c["random_connectivity_max"]), 100 * (1 - c["random_connectivity_min"])
        deg = 100 * (1 - c["topdegree_connectivity"])
        ax_c.plot([rand, r["loss"]], [yi, yi], color=GRID, lw=1.0, zorder=1)
        ax_c.plot([rand_lo, rand_hi], [yi, yi], color=INK_2, lw=1.0, zorder=2)
        ax_c.plot(rand, yi, "o", ms=3.6, mfc="white", mec=INK_2, mew=1.0, zorder=3)
        ax_c.plot(deg, yi, marker="D", ms=3.0, ls="", color=MUTED, zorder=3)
        ax_c.plot(r["loss"], yi, "o", ms=4.6, color=FAMILY_COLORS[r["family"]], mec="white", mew=0.5, zorder=4)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([f"R{r['role']} {ROLE_NAMES_ISRAEL[r['role']]} ({100 * r['share']:.1f}%, {r['seed_count']} designated)" for r in order])
    ax_c.set_xlim(-3, 105)
    ax_c.set_xlabel("connectivity loss at the same removal budget (%)")
    ax_c.tick_params(axis="y", length=0)
    light_grid(ax_c, axis="x")
    handles = [
        Line2D([], [], marker="o", ls="", ms=4.6, color=NEUTRAL, mec="white", label="role removed"),
        Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=INK_2, mew=1.0, label="random, same budget (3 seeds)"),
        Line2D([], [], marker="D", ls="", ms=3.0, color=MUTED, label="top degree, same budget"),
    ]
    ax_c.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=3, columnspacing=1.5, labelspacing=0.3, handletextpad=0.4)

    fig.savefig(ROOT / "fig_role_landscape.pdf")
    fig.savefig(ROOT / "fig_role_landscape.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 1 (timelines)
def draw_cross_network_panel(ax, data):
    yd = np.arange(len(DATASETS))[::-1]
    for yi, (key, name, _, setting) in zip(yd, DATASETS):
        rr = role_rows(data[key])
        anc = next(r for r in rr if r["role"] == anchor_role(data[key]))
        worst = max(rr, key=lambda r: r["loss"])
        c = SETTING_COLORS[setting]
        ax.plot([anc["loss"], worst["loss"]], [yi, yi], color=c, lw=1.2, alpha=0.5, zorder=2)
        ax.plot(worst["loss"], yi, "o", ms=4.6, color=c, mec="white", mew=0.5, zorder=4)
        ax.plot(anc["loss"], yi, "o", ms=4.6, mfc="white", mec=c, mew=1.0, zorder=5)
        if worst["role"] == anc["role"]:
            if worst["loss"] > 75:
                ax.text(worst["loss"] - 3.5, yi, f"R{worst['role']} (same role)", va="center", ha="right", fontsize=6.0, color=INK_2)
            else:
                ax.text(worst["loss"] + 2.5, yi, f"R{worst['role']} (same role)", va="center", ha="left", fontsize=6.0, color=INK_2)
        else:
            ax.text(anc["loss"] + 2.5 if anc["loss"] < worst["loss"] - 12 else anc["loss"] - 2.5, yi + 0.30, f"R{anc['role']}", va="center", ha="left", fontsize=6.0, color=INK_2)
            ax.text(worst["loss"] + 2.5, yi, f"R{worst['role']}", va="center", ha="left", fontsize=6.0, color=INK_2)
    ax.set_yticks(yd)
    ax.set_yticklabels([name for _, name, _, _ in DATASETS])
    ax.set_xlim(-2, 105)
    ax.set_xlabel("connectivity loss after removing role (%)")
    ax.tick_params(axis="y", length=0)
    light_grid(ax, axis="x")
    handles = [
        Line2D([], [], marker="o", ls="", ms=4.6, mfc="white", mec=INK, mew=1.0, label="anchor-densest role"),
        Line2D([], [], marker="o", ls="", ms=4.6, color=INK, label="most damaging role"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, labelspacing=0.3, handletextpad=0.4)


def generate_timelines():
    import matplotlib.dates as mdates

    ph = load_json("phenomena.json")
    data = {key: load_json(fname) for key, _, fname, _ in DATASETS}

    fig = plt.figure(figsize=(FULL_WIDTH, 162 * MM))
    # panel d writes two annotation columns beyond its right edge, so the grid stops short
    # of the figure margin to leave room for them
    gs = fig.add_gridspec(3, 2, height_ratios=[0.86, 1.0, 1.0], width_ratios=[1.25, 1.0], hspace=0.62, wspace=0.62,
                          left=0.10, right=0.885, top=0.97, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 1])
    for lab, (fx, fy) in {"a": (0.012, 0.985), "b": (0.012, 0.695), "c": (0.012, 0.375), "d": (0.44, 0.695), "e": (0.44, 0.375)}.items():
        fig.text(fx, fy, lab, fontsize=8.0, fontweight="bold", va="top", ha="left")

    # ---- a. schematic (reuse)
    ax_a.set_axis_off()
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    col_x = [0.0, 0.245, 0.495, 0.755]
    col_w = [0.215, 0.205, 0.225, 0.245]
    headers = ["Public anchors", "USDT transfer network", "Role assignment", "Analyses"]
    for x, w, head in zip(col_x, col_w, headers):
        ax_a.text(x + w / 2, 0.985, head, ha="center", va="top", fontsize=6.8, fontweight="bold", color=INK)
    rounded_box(ax_a, col_x[0], 0.56, col_w[0], 0.32, SETTING_COLORS["sanctions"], "sanctions designations\nNBCTF (Israel): 20 orders\nOFAC: 3 programmes", fontsize=6.0)
    rounded_box(ax_a, col_x[0], 0.16, col_w[0], 0.32, SETTING_COLORS["fundraising"], "public donation wallets\nAid for Ukraine\n1 address per chain", fontsize=6.0)
    ax_a.text(col_x[0] + col_w[0] / 2, 0.10, "TRON and Ethereum, 2-hop\nneighbourhoods to 1 Jan 2025", ha="center", va="top", fontsize=6.0, color=INK_2, linespacing=1.25)
    bx, bw = col_x[1], col_w[1]
    ax_a.add_patch(FancyBboxPatch((bx, 0.16), bw, 0.70, boxstyle="round,pad=0,rounding_size=0.012", lw=0.6, edgecolor=GRID, facecolor="#FAFAFA", zorder=1))
    draw_toy_network(ax_a, bx + 0.015, 0.20, bw - 0.03, 0.62)
    ax_a.text(bx + bw / 2, 0.10, "nodes: addresses; edges: transfers\nwith value and timing", ha="center", va="top", fontsize=6.0, color=INK_2, linespacing=1.25)
    mx, mw = col_x[2], col_w[2]
    for txt, y in [("direction-aware attention encoder", 0.70), ("prototype head,\noptimal-transport targets", 0.42), ("functional roles per address\n(donor, relay, collector, hub)", 0.14)]:
        rounded_box(ax_a, mx, y, mw, 0.24, "#374151", txt, fontsize=6.0, fill_alpha="0D")
    for y in (0.66, 0.42):
        arrow(ax_a, (mx + mw / 2, y), (mx + mw / 2, y - 0.045), scale=5, lw=0.7)
    ax_a.text(mx + mw / 2, 0.10, "no curated role labels;\nanchors used only for evaluation", ha="center", va="top", fontsize=6.0, color=INK_2, linespacing=1.25)
    axx, aw = col_x[3], col_w[3]
    for txt, y in [("timing of designation relative\nto observed activity", 0.70), ("role removal versus budget-\nmatched random removal", 0.42), ("persistence of counterparties;\nsanctions versus donations", 0.14)]:
        rounded_box(ax_a, axx, y, aw, 0.24, "#374151", txt, fontsize=6.0, fill_alpha="0D")
    ax_a.text(axx + aw / 2, 0.10, "six networks, 0.6 M to 23.9 M addresses;\n2.7 M and 85 M transfers in the two largest", ha="center", va="top", fontsize=6.0, color=INK_2, linespacing=1.25)
    for i in range(3):
        arrow(ax_a, (col_x[i] + col_w[i] + 0.006, 0.52), (col_x[i + 1] - 0.006, 0.52), scale=7, lw=1.0)

    # ---- b. monthly volume through designated addresses with seizure orders
    m = ph["nbctf_monthly"]
    months = np.array([np.datetime64(x) for x in m["month"]])
    vol = np.array(m["designated_volume_usdt"]) / 1e6
    ax_b.plot(months, np.maximum(vol, 1e-3), color=SETTING_COLORS["sanctions"], lw=1.2, marker="o",
              ms=2.0, mec="white", mew=0.3, zorder=3)
    ax_b.fill_between(months, 1e-2, np.maximum(vol, 1e-3), color=SETTING_COLORS["sanctions"],
                      alpha=0.12, lw=0, zorder=2)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.01, 2.0e7)
    ax_b.set_ylabel("USDT through designated\naddresses per month (million)")
    i = 0
    for o in ph["nbctf_orders"]:
        d = np.datetime64(o["signed"])
        if d > np.datetime64("2025-01-01") or o["n_addresses"] < 3:
            continue
        ax_b.axvline(d, color=INK_2, lw=0.6, ls=(0, (2, 2)), zorder=2)
        if o["n_addresses"] >= 10:
            ax_b.text(d + np.timedelta64(22 * (1 if i % 2 else -1), 'D'), (3.0e3, 3.0e4, 3.0e5, 3.0e6)[i % 4], f"{o['order'].replace('ASO ', '')} ({o['n_addresses']})", rotation=90, ha="center", va="bottom", fontsize=6.0, color=INK_2,
                      bbox=dict(fc="white", ec="none", pad=0.8))
            i += 1
    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_b.set_xlim(np.datetime64("2020-01-01"), np.datetime64("2025-01-15"))
    light_grid(ax_b, axis="y")

    # ---- c. Ukraine TRON donations, cumulative on a log time axis
    # A thousand daily bars in 66 mm are mostly narrower than a pixel, and the claim the panel
    # supports is cumulative: the campaign was over in its first week. Log days since the
    # invasion gives that week a third of the axis instead of a third of a millimetre.
    u = ph["ukraine"]["tron"]
    days = np.array([np.datetime64(x) for x in u["daily"]["date"]])
    t0 = np.datetime64("2022-02-24")
    el = (days - t0).astype("timedelta64[D]").astype(float) + 1.0
    keep = el >= 1
    el = el[keep]
    cv = np.cumsum(np.array(u["daily"]["volume_usdt"])[keep]) / u["volume_usdt"] * 100
    cn = np.cumsum(np.array(u["daily"]["donations"])[keep]) / u["n_donations"] * 100
    ax_c.step(el, cv, where="post", color=SETTING_COLORS["fundraising"], lw=1.4, zorder=4,
              label="share of USDT donated")
    ax_c.step(el, cn, where="post", color=INK_2, lw=1.1, ls=(0, (3, 1.6)), zorder=3,
              label="share of donations made")
    ax_c.axvspan(1, 7, color=SETTING_COLORS["fundraising"], alpha=0.10, lw=0, zorder=1)
    w1 = u["windows"]["week1"]
    ax_c.text(1.15, 46,
              f"first week:\n{w1['volume_usdt']/1e6:.2f} M USDT from {w1['donors']:,} donors",
              fontsize=6.0, color=INK_2, va="bottom", linespacing=1.25)
    ax_c.set_xscale("log")
    ax_c.set_xlim(1, 1100)
    ax_c.set_ylim(0, 104)
    ax_c.set_xlabel("days since the invasion of 24 February 2022 (log scale)")
    ax_c.set_ylabel("cumulative share of the\ncampaign's total (%)")
    ax_c.set_xticks([1, 7, 30, 90, 365, 1000])
    ax_c.set_xticklabels(["1", "7", "30", "90", "365", "1,000"])
    ax_c.text(0.02, 0.97, f"total: {u['volume_usdt']/1e6:.2f} M USDT\n{u['n_donations']:,} donations from {u['n_donors']:,} donors",
              transform=ax_c.transAxes, fontsize=6.0, color=MUTED, ha="left", va="top", linespacing=1.25,
              bbox=dict(fc="white", ec="none", pad=0.8))
    ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=False, fontsize=6.0,
                handlelength=1.8, labelspacing=0.3)
    light_grid(ax_c, axis="y")

    # ---- d. network scale
    y = np.arange(len(DATASETS))[::-1]
    nodes = np.array([data[k]["n_nodes"] for k, *_ in DATASETS], dtype=float)
    edges = np.array([unique_edges(k, data) for k, *_ in DATASETS], dtype=float)
    seeds = [sum(r["seed_count"] for r in role_rows(data[k])) for k, *_ in DATASETS]
    ks = [data[k]["n_roles"] for k, *_ in DATASETS]
    for yi, n, e, (key, name, _, setting) in zip(y, nodes, edges, DATASETS):
        c = SETTING_COLORS[setting]
        ax_d.plot([n, e], [yi, yi], color=c, lw=1.2, alpha=0.5, zorder=2)
        ax_d.plot(n, yi, "o", ms=4.2, color=c, mec="white", mew=0.5, zorder=3)
        ax_d.plot(e, yi, "o", ms=4.2, mfc="white", mec=c, mew=1.0, zorder=3)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([name for _, name, _, _ in DATASETS])
    ax_d.set_xscale("log")
    ax_d.set_xlim(4e3, 2e8)
    ax_d.set_xlabel("count (log scale)")
    light_grid(ax_d, axis="x")
    ax_d.tick_params(axis="y", length=0)
    for yi, k, s_ in zip(y, ks, seeds):
        ax_d.text(1.02, yi, f"{k}", transform=ax_d.get_yaxis_transform(), ha="left", va="center", fontsize=6.0, color=INK_2)
        ax_d.text(1.15, yi, f"{s_}", transform=ax_d.get_yaxis_transform(), ha="left", va="center", fontsize=6.0, color=INK_2)
    ax_d.text(1.02, 1.01, "K", transform=ax_d.transAxes, ha="left", va="bottom", fontsize=6.0, color=INK, fontweight="bold")
    ax_d.text(1.15, 1.01, "anchors", transform=ax_d.transAxes, ha="left", va="bottom", fontsize=6.0, color=INK, fontweight="bold")
    scale_handles = [
        Line2D([], [], marker="o", ls="", ms=4.2, color=NEUTRAL, mec="white", label="addresses"),
        Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec=NEUTRAL, mew=1.0, label="unique directed edges"),
    ]
    ax_d.legend(handles=scale_handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.4, labelspacing=0.3)

    # ---- e. role composition
    focal = [("israel", "NBCTF Israel\n(TRON)"), ("ukr_tron", "Ukraine aid\n(TRON)"), ("ukr_eth", "Ukraine aid\n(ETH)")]
    yc = np.arange(len(focal))[::-1]
    for yi, (key, name) in zip(yc, focal):
        rows = role_rows(data[key])
        anchor = anchor_role(data[key])
        left = 0.0
        for r in rows:
            width = r["share"] * 100
            ax_e.barh(yi, width, left=left, height=0.55, color=FAMILY_COLORS[r["family"]], edgecolor="white", linewidth=0.6, zorder=2)
            if width > 5.5:
                ax_e.text(left + width / 2, yi, f"R{r['role']}", ha="center", va="center", fontsize=6.0, color="white", fontweight="bold", zorder=3)
            if r["role"] == anchor:
                ax_e.plot(left + width / 2, yi + 0.42, marker="v", ms=3.5, color=INK, zorder=4, clip_on=False)
            left += width
    ax_e.set_yticks(yc)
    ax_e.set_yticklabels([name for _, name in focal])
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(-0.6, len(focal) - 0.2)
    ax_e.set_xlabel("share of addresses (%)")
    light_grid(ax_e, axis="x")
    ax_e.tick_params(axis="y", length=0)
    fam_handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=FAMILY_LABELS[f]) for f in FAMILY_ORDER]
    fam_handles.append(Line2D([], [], marker="v", ls="", ms=3.5, color=INK, label="role with most anchor addresses"))
    fig.legend(handles=fam_handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, columnspacing=1.2, handletextpad=0.5, handlelength=1.0, handleheight=0.8, labelspacing=0.35)

    fig.savefig(ROOT / "fig_study_design.pdf")
    fig.savefig(ROOT / "fig_study_design.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 2 (designation timing)
def generate_designation():
    ph = load_json("phenomena.json")
    t, e, p, c = ph["nbctf_timing"], ph["nbctf_event_study"], ph["placebo_event_study"], ph["counterparty_persistence"]
    W = ph["window_weeks"]
    weeks = np.arange(-W, W + 1)

    fig = plt.figure(figsize=(FULL_WIDTH, 110 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.46, left=0.135, right=0.975, top=0.955, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    for ax, lab, dx in zip([ax_a, ax_b, ax_c, ax_d], "abcd", (-0.34, -0.20, -0.20, -0.20)):
        panel_label(ax, lab, dx=dx)

    # ---- a. lag to signing, one row per order
    # The unit of assignment is the order, not the address, and the order medians run from
    # -279 to +163 days. A pooled histogram hides exactly the variation the text is about.
    by = t["days_last_activity_to_signed_by_order"]
    signed = t["signed_by_order"]
    orders = sorted(by, key=lambda o: signed[o])
    dd = np.array(t["days_last_activity_to_signed"])
    rng = np.random.default_rng(7)
    y = np.arange(len(orders))[::-1]
    for yi, o in zip(y, orders):
        v = np.array(by[o])
        ax_a.scatter(v, yi + rng.uniform(-0.22, 0.22, len(v)), s=3.2, lw=0,
                     color=SETTING_COLORS["sanctions"], alpha=0.55, zorder=3)
        m = float(np.median(v))
        ax_a.plot([m, m], [yi - 0.34, yi + 0.34], color=INK, lw=1.2, zorder=5,
                  solid_capstyle="butt")
    ax_a.axvline(0, color=INK_2, lw=0.7, zorder=2)
    med = float(np.median(dd))
    ax_a.axvline(med, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{o}  ($n$ = {len(by[o])})" for o in orders], fontsize=6.0)
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_ylim(-0.7, len(orders) - 0.3)
    ax_a.set_xlim(-700, 640)
    ax_a.set_xlabel("days from last observed transfer to signing of the seizure order")
    ax_a.text(-685, len(orders) - 0.42, "still active after signing", fontsize=6.0, color=MUTED, va="bottom")
    ax_a.text(625, len(orders) - 0.42, "dormant before signing", fontsize=6.0, color=MUTED,
              va="bottom", ha="right")
    ax_a.text(med + 14, -0.52, f"pooled median {med:.1f} d", fontsize=6.0, color=INK, va="center")
    light_grid(ax_a, axis="x")

    # ---- b. weekly volume normalised to pre-event mean
    ev = np.array(e["volume_usdt"]) / e["pre_mean_weekly_volume"]
    pv = np.array(p["volume_usdt"]) / p["pre_mean_weekly_volume"]
    ax_b.axvspan(-0.5, 0.5, color=GRID, zorder=1)
    ax_b.plot(weeks, np.maximum(pv, 6e-3), color=MUTED, lw=1.0, marker="o", ms=2.2, mec="white", mew=0.3, zorder=3, label=f"placebo: undesignated counterparties (n = {p['n_addresses']:,})")
    ax_b.plot(weeks, np.maximum(ev, 6e-3), color=SETTING_COLORS["sanctions"], lw=1.4, marker="o", ms=2.6, mec="white", mew=0.3, zorder=4, label=f"designated addresses (n = {e['n_addresses']})")
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.005, 5)
    ax_b.set_xlabel("weeks relative to signing of the seizure order")
    ax_b.set_ylabel("weekly USDT volume\n(relative to pre-signing mean; floor = 0.006)")
    ax_b.axhline(1, color=INK_2, lw=0.5, ls=(0, (2, 2)), zorder=2)
    light_grid(ax_b)
    ax_b.legend(loc="lower left", labelspacing=0.3, handlelength=1.6)

    # ---- c. share of addresses active by relative week
    ea = np.array(e["per_address"]["share_active_by_week"]) * 100
    pa = np.array(p["per_address"]["share_active_by_week"]) * 100
    ax_c.axvspan(-0.5, 0.5, color=GRID, zorder=1)
    ax_c.plot(weeks, pa, color=MUTED, lw=1.0, marker="o", ms=2.2, mec="white", mew=0.3, zorder=3, label="placebo")
    ax_c.plot(weeks, ea, color=SETTING_COLORS["sanctions"], lw=1.4, marker="o", ms=2.6, mec="white", mew=0.3, zorder=4, label="designated")
    ax_c.set_xlabel("weeks relative to signing of the seizure order")
    ax_c.set_ylabel("addresses with at least\none transfer in the week (%)")
    ax_c.set_ylim(0, max(ea.max(), pa.max()) * 1.25)
    light_grid(ax_c)
    ax_c.legend(loc="upper right", labelspacing=0.3, handlelength=1.6)

    # ---- d. persistence after designation
    groups = [
        ("designated\naddresses", [t["share_active_after_signing"] * 100, t["volume_after_signing_share"] * 100], SETTING_COLORS["sanctions"]),
        ("their undesignated\ncounterparties", [c["share_active_after_order"] * 100, c["volume_share_after_order"] * 100], INK_2),
    ]
    x = np.arange(2)
    wbar = 0.36
    for i, (lab, vals, col) in enumerate(groups):
        ax_d.bar(x + (i - 0.5) * wbar, vals, width=wbar - 0.04, color=col, alpha=0.9, zorder=3, label=lab.replace("\n", " "))
        for xi, v in zip(x + (i - 0.5) * wbar, vals):
            ax_d.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.0, color=INK)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["share of addresses with any\ntransfer after the order", "share of the address's USDT\nvolume occurring after the order"])
    ax_d.set_ylabel("per cent")
    ax_d.set_ylim(0, 100)
    light_grid(ax_d, axis="y")
    ax_d.legend(loc="upper left", labelspacing=0.3, handlelength=1.0, handleheight=0.8)
    ax_d.text(0.99, 0.97, f"{c['share_active_90d_after_order']*100:.0f}% of counterparties still active\n90 days after the order", transform=ax_d.transAxes, fontsize=6.0, color=INK_2, ha="right", va="top", linespacing=1.25)

    fig.savefig(ROOT / "fig_designation.pdf")
    fig.savefig(ROOT / "fig_designation.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 4 (backbone and concentration)
def draw_removal_columns(axes):
    """Three aligned columns over the same removal sets: addresses, throughput, stranded value.

    The three measure different things and the paper turns on the difference. Addresses lost
    counts accounts that lose their only route. Throughput is the value the removed set moved
    itself, which says how large it was. Value stranded is USDT that was moving between two
    addresses that both survive and that the removal separates, which is the only one of the
    three that says the removed set carried anything for anyone else.
    """
    fn = load_json("full_tron_backbone.json")
    fs = load_json("full_tron_stranded.json")["decomposition"]
    dmi = load_json("degree_matched_interval.json")
    na = fn["n_designated"]
    iso, thr = dmi["isolated_share_pct_summary"], dmi["throughput_pct_summary"]
    strand_draws = dmi.get("stranded_usdt_exact_draws", [])

    rows = [
        (f"the {na} designated", SETTING_COLORS["sanctions"],
         (fn["remove_designated_pct"], None), (fs["designated"]["incident_pct"], None),
         (fs["designated"]["stranded_usdt"], None)),
        (f"{na} undesignated, degree-matched", NEUTRAL,
         (iso["mean"], iso["ci"]), (thr["mean"], thr["ci"]),
         (None, sorted(strand_draws) if strand_draws else None)),
        (f"{na} undesignated, at random", MUTED,
         (fn["remove_random_pct_mean"], None), (None, None), (None, None)),
        (f"{na} undesignated, highest degree", INK_2,
         (fn["remove_top_degree_undesignated_pct"], None), (fs[f"top_degree_{na}"]["incident_pct"], None),
         (fs[f"top_degree_{na}"]["stranded_usdt"], None)),
        ("1,000 undesignated, highest degree", INK_2,
         (fn["remove_top_degree_undesignated_1000_pct"], None), (fs["top_degree_1000"]["incident_pct"], None),
         (fs["top_degree_1000"]["stranded_usdt"], None)),
        ("10,000 undesignated, highest degree", INK_2,
         (fn["remove_top_degree_undesignated_10000_pct"], None), (fs["top_degree_10000"]["incident_pct"], None),
         (fs["top_degree_10000"]["stranded_usdt"], None)),
    ]
    y = np.arange(len(rows))[::-1]
    specs = [(0, "addresses lost from the\nlargest component (%)", 1e-5, 130, "pct"),
             (1, "throughput of the\nremoved set (% of value)", 5e-3, 130, "pct"),
             (2, "USDT stranded between\nsurviving addresses", 1e2, 3e11, "usdt")]
    for col, xlabel, lo, hi, kind in specs:
        ax = axes[col]
        for yi, r in zip(y, rows):
            v, ci = r[2 + col]
            c = r[1]
            if v is None and ci is not None:
                for x in ci:
                    ax.plot(max(x, lo * 1.25), yi, "o", ms=3.4, color=c, mec="white", mew=0.5, zorder=4)
                ax.text(max(ci[-1], lo * 1.25) * 1.55, yi, "  ".join(f"{x:,.0f}" for x in ci),
                        fontsize=6.0, color=INK_2, va="center")
                continue
            if v is None:
                ax.text(lo * 1.6, yi, "not computed", fontsize=6.0, color=MUTED, va="center")
                continue
            vv = max(v, lo * 1.25)
            if ci is not None:
                ax.plot([max(ci[0], lo * 1.25), ci[1]], [yi, yi], color=c, lw=1.0, zorder=3)
            edge = "<" if v < lo else "o"
            ax.plot(vv, yi, "o" if edge == "o" else "<", ms=4.0, color=c, mec="white", mew=0.6, zorder=4)
            if kind == "pct":
                txt = ("$<10^{-4}$" if v < 1e-4 else (f"{v:.4f}" if v < 0.1 else f"{v:.1f}"))
            else:
                txt = (f"{v:,.0f}" if v < 1e6 else f"{v/1e9:.2f} bn")
            at = ci[1] if ci is not None else vv
            ax.text(at * 1.55, yi, txt, fontsize=6.0, color=INK_2, va="center")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_xlabel(xlabel, fontsize=6.0, linespacing=1.35)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if col == 0 else [])
        ax.tick_params(axis="y", length=0)
        light_grid(ax, axis="x")
    axes[0].text(0.0, 1.045, "complete TRON USDT network: 213M addresses, 745M directed pairs, 15.5 trillion USDT",
                 transform=axes[0].transAxes, fontsize=6.0, color=INK, va="bottom")


def draw_boundary_panel(ax):
    """Where the crawl stops decides the answer, and the answer reverses.

    One dumbbell per network and boundary: the orange dot is the loss from removing the
    designated addresses, the grey dot from removing the same number of undesignated hubs.
    The orange dot is on the left at two hops and on the right at one hop, in every network.
    """
    bs = load_json("boundary_sensitivity.json")
    fn = load_json("full_tron_backbone.json")
    names = [("israel_tron", "NBCTF Israel"), ("ofac_iran_tron", "OFAC Iran"),
             ("ofac_russia_ukraine_tron", "OFAC Russia\u2013Ukr."),
             ("ofac_terrorist_financing_tron", "OFAC terrorism")]
    labels, des, top, kinds = [], [], [], []
    for tag, name in names:
        for key, gl in (("two_hop", "two hops"), ("hop1", "one hop")):
            labels.append(f"{name}, {gl}")
            des.append(bs[tag][key]["remove_designated_pct"])
            top.append(bs[tag][key]["remove_top_degree_undesignated_pct"])
            kinds.append(gl)
    labels.append("complete network, no boundary")
    des.append(fn["remove_designated_pct"])
    top.append(fn["remove_top_degree_undesignated_pct"])
    kinds.append("complete")

    y = np.arange(len(labels))[::-1]
    for yi, d, t, k in zip(y, des, top, kinds):
        ax.plot([d, t], [yi, yi], color=GRID, lw=1.4, zorder=2, solid_capstyle="round")
        ax.plot(t, yi, "o", ms=4.0, color=INK_2, mec="white", mew=0.6, zorder=4)
        ax.plot(d, yi, "o", ms=4.0, color=SETTING_COLORS["sanctions"], mec="white", mew=0.6, zorder=5)
    for yi, k in zip(y, kinds):
        if k == "complete":
            ax.axhline(yi + 0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-4, 100)
    ax.set_xlabel("connectivity loss (%)")
    light_grid(ax, axis="x")
    handles = [Line2D([], [], marker="o", ls="", ms=4.0, color=SETTING_COLORS["sanctions"],
                      label="the designated addresses removed"),
               Line2D([], [], marker="o", ls="", ms=4.0, color=INK_2,
                      label="the same number of undesignated hubs removed")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.50, -0.24),
              frameon=False, fontsize=6.0, handlelength=1.0, labelspacing=0.3)


def generate_backbone():
    ph = load_json("phenomena.json")
    k, u = ph["concentration"], ph["ukraine"]["tron"]

    fig = plt.figure(figsize=(FULL_WIDTH, 106 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.10], height_ratios=[1.0, 1.32],
                          wspace=0.30, hspace=0.55, left=0.235, right=0.975, top=0.945, bottom=0.155)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_a3 = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c = fig.add_subplot(gs[1, 2])
    panel_label(ax_a1, "a", dx=-0.62)
    panel_label(ax_b, "b", dx=-0.30)
    panel_label(ax_c, "c", dx=-0.28)

    draw_removal_columns([ax_a1, ax_a2, ax_a3])
    draw_boundary_panel(ax_b)

    # ---- c. concentration of volume among counterparties and among donors
    # ranks are log-spaced so the head of the distribution, where the quantities the text
    # quotes live, is actually drawn rather than interpolated across
    cx = np.array(k["counterparty_lorenz_log_rank"]) / k["n_counterparties_ranked"] * 100
    ux = np.array(u["donor_lorenz_log_rank"]) / u["n_donors_ranked"] * 100
    ax_c.plot(cx, np.array(k["counterparty_lorenz_log"]) * 100, color=SETTING_COLORS["sanctions"], lw=1.4,
              zorder=3, label=f"counterparties of designated addresses ($n$ = {k['n_counterparties']:,})")
    ax_c.plot(ux, np.array(u["donor_lorenz_log"]) * 100, color=SETTING_COLORS["fundraising"], lw=1.4,
              zorder=3, label=f"donors to the Ukraine TRON address ($n$ = {u['n_donors']:,})")
    ax_c.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax_c.text(1.18, 3, "top 1% of addresses", fontsize=6.0, color=INK_2, rotation=90, va="bottom")
    ax_c.set_xscale("log")
    ax_c.set_xlim(1 / k["n_counterparties_ranked"] * 100 * 0.7, 100)
    ax_c.set_ylim(0, 103)
    ax_c.set_xlabel("top share of addresses by volume (%)")
    ax_c.set_ylabel("share of USDT volume (%)")
    light_grid(ax_c)
    ax_c.text(0.85, k["top1pct_share"] * 100 - 9, f"{k['top1pct_share']*100:.0f}%", fontsize=6.0,
              color=SETTING_COLORS["sanctions"], ha="right",
              bbox=dict(fc="white", ec="none", pad=0.6))
    ax_c.text(0.85, u["top1pct_donor_volume_share"] * 100 + 3, f"{u['top1pct_donor_volume_share']*100:.0f}%",
              fontsize=6.0, color=SETTING_COLORS["fundraising"], ha="right",
              bbox=dict(fc="white", ec="none", pad=0.6))
    ax_c.legend(loc="upper left", bbox_to_anchor=(-0.30, -0.24), frameon=False, fontsize=6.0,
                labelspacing=0.3, handlelength=1.6)

    fig.savefig(ROOT / "fig_backbone.pdf")
    fig.savefig(ROOT / "fig_backbone.png", dpi=300)
    plt.close(fig)
    print("wrote fig_backbone")


def generate_donation_sizes_si():
    """Donation-size distribution, moved out of the main text: it supports one sentence."""
    u = load_json("phenomena.json")["ukraine"]["tron"]
    fig, ax = plt.subplots(figsize=(88 * MM, 58 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.94, bottom=0.20)
    h = u["donation_size_hist"]
    edges_ = np.array(h["bin_edges_usdt"]); counts = np.array(h["counts"])
    centers = np.sqrt(edges_[:-1] * edges_[1:])
    ax.bar(centers, counts, width=np.diff(edges_) * 0.9, color=SETTING_COLORS["fundraising"],
           alpha=0.85, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2e5)
    ax.set_xlabel("donation size (USDT)")
    ax.set_ylabel("number of donations")
    ax.axvline(u["median_donation_usdt"], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.text(u["median_donation_usdt"] * 1.5, counts.max() * 0.9,
            f"median {u['median_donation_usdt']:.0f} USDT", fontsize=6.0, color=INK, va="top")
    ax.text(0.98, 0.72, f"{u['share_donations_below_100']*100:.0f}% of donations\nbelow 100 USDT",
            transform=ax.transAxes, fontsize=6.0, color=INK_2, ha="right", va="top", linespacing=1.25)
    light_grid(ax, axis="y")
    fig.savefig(ROOT / "fig_donation_sizes_si.pdf")
    fig.savefig(ROOT / "fig_donation_sizes_si.png", dpi=300)
    plt.close(fig)
    print("wrote fig_donation_sizes_si")


def generate_cross_network_si():
    data = {key: load_json(fname) for key, _, fname, _ in DATASETS}
    fig = plt.figure(figsize=(88 * MM, 62 * MM))
    ax = fig.add_axes([0.42, 0.16, 0.55, 0.80])
    draw_cross_network_panel(ax, data)
    fig.savefig(ROOT / "fig_cross_network_si.pdf")
    fig.savefig(ROOT / "fig_cross_network_si.png", dpi=300)
    plt.close(fig)


def generate_enforcement():
    ph = load_json("phenomena.json")
    t = ph["tether_enforcement"]
    W = ph["window_weeks"]
    weeks = np.arange(-W, W + 1)

    fig = plt.figure(figsize=(FULL_WIDTH, 124 * MM))
    gs = fig.add_gridspec(2, 3, hspace=1.15, wspace=0.50, left=0.075, right=0.985, top=0.955,
                          bottom=0.10, width_ratios=[1.0, 0.86, 0.86])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    for ax, lab, dx in zip([ax_a, ax_b, ax_c, ax_d, ax_e], "abcde",
                           (-0.11, -0.26, -0.26, -0.26, -0.26)):
        panel_label(ax, lab, dx=dx)

    # ---- a. freeze coverage by order (stacked bars)
    orders = [o for o in t["by_order"] if o["n"] >= 3]
    x = np.arange(len(orders))
    cats = [("frozen before signing", INK), ("frozen within 30 d after", INK_2),
            ("frozen later", NEUTRAL), ("never frozen", "#FFFFFF")]
    fr = load_json("phenomena.json")["tether_enforcement"]
    # per-order breakdown requires the per-address list; recompute from days list is not per order, so use by_order fields
    bottoms = np.zeros(len(orders))
    for ci, (lab, col) in enumerate(cats):
        vals = []
        for o in orders:
            before = o["n_frozen_before_signing"]
            within = o.get("n_frozen_within_30d", 0)
            later = o["n_frozen"] - before - within
            never = o["n"] - o["n_frozen"]
            v = [before, within, later, never][ci]
            vals.append(100 * v / o["n"])
        vals = np.array(vals)
        ax_a.bar(x, vals, bottom=bottoms, color=col, width=0.7,
                 edgecolor=(INK_2 if col == "#FFFFFF" else "white"), linewidth=0.5, label=lab, zorder=3)
        bottoms += vals
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{o['signed'][2:10]} (n={o['n']})" for o in orders], fontsize=6.0, rotation=90)
    ax_a.set_ylabel("designated addresses (%)")
    ax_a.set_ylim(0, 100)
    light_grid(ax_a, axis="y")
    ax_a.set_xlabel("seizure order (signing date, addresses with an observed transfer)")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.62), ncol=2, columnspacing=0.8,
                handlelength=1.0, handleheight=0.8, labelspacing=0.3, fontsize=6.0, frameon=False)

    # ---- b. days from signing to Tether freeze
    dd = np.array(t["days_signed_to_frozen"])
    bins = np.arange(-100, 401, 10)
    ax_b.hist(np.clip(dd, -99, 399), bins=bins, color=INK_2, alpha=0.85, zorder=3)
    ax_b.axvline(0, color=INK_2, lw=0.7, zorder=4)
    ax_b.axvline(np.median(dd), color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax_b.text(np.median(dd) + 14, ax_b.get_ylim()[1] * 0.60, f"median {np.median(dd):.0f} d", fontsize=6.0, color=INK, va="top")
    ax_b.text(0.99, 0.97, f"{t['n_frozen']} of {t['n_designated']} frozen\n{t['n_frozen_before_signing']} before signing\n{t['n_frozen_within_30d']} within 30 d after\n{t['n_frozen_after_30d']} later", transform=ax_b.transAxes, fontsize=6.0, color=INK_2, ha="right", va="top", linespacing=1.3)
    ax_b.set_xlabel("days from signing of the order\nto Tether blacklisting", linespacing=1.35)
    ax_b.set_ylabel("frozen designated addresses")
    ax_b.set_xlim(-100, 400)
    light_grid(ax_b, axis="y")

    # ---- c. weekly inflow/outflow relative to the freeze
    e = t["event_study_freeze"]
    vi = np.array(e["inflow_usdt"]) / 1e6
    vo = np.array(e["outflow_usdt"]) / 1e6
    m = (weeks >= -26) & (weeks <= 4)
    ax_c.axvspan(-0.5, 0.5, color=GRID, zorder=1)
    ax_c.plot(weeks[m], vi[m], color=INK_2, lw=1.1, ls=(0, (3, 1.6)), marker="o", ms=2.4, mec="white", mew=0.3, zorder=3, label="inflow to frozen addresses")
    ax_c.plot(weeks[m], vo[m], color=SETTING_COLORS["sanctions"], lw=1.4, marker="o", ms=2.4, mec="white", mew=0.3, zorder=4, label="outflow from frozen addresses")
    ax_c.set_xlabel("weeks relative to Tether blacklisting")
    ax_c.set_ylabel(f"USDT per week (million)\n$n$ = {e['n_addresses']} frozen addresses", linespacing=1.35)
    ax_c.set_ylim(0, max(vi[m].max(), vo[m].max()) * 1.25)
    light_grid(ax_c)
    ax_c.legend(loc="upper left", labelspacing=0.3, handlelength=1.6)

    # ---- d. dormancy at freeze and balance
    dl = np.array([x for x in ph["tether_enforcement"].get("days_last_transfer_to_freeze", []) if x is not None])
    if len(dl):
        bins = np.arange(-50, 801, 25)
        ax_d.hist(np.clip(dl, -49, 799), bins=bins, color=INK_2, alpha=0.85, zorder=3)
        ax_d.axvline(np.median(dl), color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
        ax_d.text(np.median(dl) + 10, ax_d.get_ylim()[1] * 0.95, f"median {np.median(dl):.0f} d", fontsize=6.0, color=INK, va="top")
    ax_d.set_xlabel("days from the address's last transfer to Tether blacklisting")
    ax_d.set_ylabel("frozen designated addresses")
    ax_d.text(0.99, 0.82, f"{t['share_of_frozen_addresses_with_positive_balance']*100:.0f}% held more than\n1 USDT when frozen", transform=ax_d.transAxes, fontsize=6.0, color=INK_2, ha="right", va="top", linespacing=1.25)
    light_grid(ax_d, axis="y")

    # ---- e. what was still there when the lock closed
    # Every address is one point: everything it ever received against what remained at the
    # freeze. The y = x line is the seizure the order asks for; the cloud sits four to six
    # decades below it, which is the paragraph this figure is under.
    inflow = np.array(t["per_address_lifetime_inflow_usdt"])
    bal = np.array(t["per_address_balance_at_freeze_usdt"])
    ok = inflow > 0
    lo, hi = 1e-3, 1e10
    xs = inflow[ok]
    ys = np.clip(bal[ok], lo, None)
    zero = bal[ok] < 1e-6
    ax_e.plot([lo, hi], [lo, hi], color=INK_2, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax_e.scatter(xs[~zero], ys[~zero], s=3.4, lw=0, color=SETTING_COLORS["sanctions"],
                 alpha=0.55, zorder=4)
    ax_e.scatter(xs[zero], np.full(int(zero.sum()), lo * 1.6), s=3.4, lw=0, color=MUTED,
                 alpha=0.55, marker="v", zorder=3)
    ax_e.set_xscale("log"); ax_e.set_yscale("log")
    ax_e.set_xlim(1e0, hi); ax_e.set_ylim(lo * 0.5, hi)
    ax_e.set_xlabel("USDT ever received by the address")
    ax_e.set_ylabel("USDT still held when frozen")
    ax_e.text(6e3, 1.2e5, "everything received\nstill present", fontsize=6.0, color=INK_2,
              rotation=39, va="bottom", linespacing=1.2)
    ax_e.text(0.03, 0.99, f"{t['balance_at_freeze_total_usdt']/1e6:.1f} M of "
              f"{t['lifetime_inflow_frozen_usdt']/1e9:.2f} bn USDT\nfrozen ("
              f"{100*t['balance_at_freeze_total_usdt']/t['lifetime_inflow_frozen_usdt']:.2f}%)",
              transform=ax_e.transAxes, fontsize=6.0, color=INK_2, ha="left", va="top",
              linespacing=1.25)
    ax_e.text(1.6e0, lo * 2.2, "nothing left", fontsize=6.0, color=MUTED, va="bottom")
    light_grid(ax_e)

    fig.savefig(ROOT / "fig_enforcement.pdf")
    fig.savefig(ROOT / "fig_enforcement.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    generate_timelines()
    generate_designation()
    generate_enforcement()
    generate_role_landscape()
    generate_backbone()
    generate_donation_sizes_si()
    generate_cross_network_si()
    generate_model_performance()
