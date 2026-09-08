#!/usr/bin/env python3
"""Generate Supplementary Information tables directly from the result JSON files.

Every number in these tables is produced by the same JSONs that feed the main
figures, so the supplement cannot drift from the figures. Outputs (LaTeX
fragments, \\input into online_appendix.tex):
  table_ofac_datasets.tex        network sizes and in-graph anchors for all analysed networks
  table_ofac_roles.tex           per-role profiles for the three OFAC TRON networks
  table_cross_regime.tex         structural position of the anchor-densest role per network
  table_dismantling_controls.tex role removal versus budget-matched random / top-degree removal
  table_screening_extended.tex   IC coverage versus centralities on all / hop-1 / hop-2 candidate sets
  table_recall_at_k.tex          recall@K on the primary benchmark
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_ncomms_figures import DATASETS, anchor_role, load_json, role_rows  # noqa: E402

OFAC = [
    ("ofac_iran", "OFAC Iran (TRON)"),
    ("ofac_ru", "OFAC Russia--Ukraine (TRON)"),
    ("ofac_tf", "OFAC terrorist financing (TRON)"),
]
FAMILY_TEX = {"sender": "send-dominant", "receiver": "receive-dominant", "relay": "balanced low-degree", "hub": "high-degree bilateral"}
SHORT_FAMILY = {"sender": "sender", "receiver": "receiver", "relay": "relay", "hub": "hub"}


def tex_escape(s: str) -> str:
    return s.replace("–", "--").replace("%", "\\%")


def fmt_int(x) -> str:
    return f"{int(x):,}"


def write(name: str, body: str) -> None:
    (ROOT / name).write_text(body)
    print(f"wrote {name}")


def main() -> None:
    data = {key: load_json(fname) for key, _, fname, _ in DATASETS}
    names = {key: tex_escape(name) for key, name, _, _ in DATASETS}
    controls = json.load(open(ROOT / "dismantling_controls.json")) if (ROOT / "dismantling_controls.json").exists() else None
    screening = json.load(open(ROOT / "screening_extended.json")) if (ROOT / "screening_extended.json").exists() else None

    # ---- network overview
    rows = []
    for key, name, _, setting in DATASETS:
        d = data[key]
        n_edges = controls[{"israel": "israel_tron", "ukr_tron": "ukraine_tron", "ukr_eth": "ukraine_eth", "ofac_iran": "ofac_iran_tron", "ofac_ru": "ofac_russia_ukraine_tron", "ofac_tf": "ofac_terrorist_financing_tron"}[key]]["n_edges"] if controls else d["n_edges"]
        anchors = sum(r["seed_count"] for r in role_rows(d))
        rows.append(f"{tex_escape(name)} & {setting} & {fmt_int(d['n_nodes'])} & {fmt_int(n_edges)} & {d['n_roles']} & {anchors} \\\\")
    write("table_ofac_datasets.tex", "\\begin{tabular}{llrrrr}\n\\toprule\n\\textbf{Network} & \\textbf{Anchor type} & \\textbf{Addresses} & \\textbf{Unique directed edges} & $K$ & \\textbf{Anchors in graph} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- primary and Ukraine role profiles
    for key, name, fname in [("israel", "NBCTF (Israel) TRON USDT", "table_israel_roles.tex"), ("ukr_tron", "Aid for Ukraine TRON USDT", "table_ukraine_tron_roles.tex"), ("ukr_eth", "Aid for Ukraine Ethereum USDT", "table_ukraine_eth_roles.tex")]:
        d = data[key]
        anchor = anchor_role(d)
        lines = []
        for r in role_rows(d):
            star = "$^*$" if r["role"] == anchor else ""
            ic = d["role_stats"][str(r["role"])].get("avg_ic")
            lines.append(f"R{r['role']}{star} & {FAMILY_TEX[r['family']]} & {fmt_int(r['size'])} & {100*r['share']:.1f} & {r['avg_in']:.2f} & {r['avg_out']:.2f} & {ic if ic is not None else float('nan'):.4f} & {r['seed_count']} & {r['seed_density']:.3f} & {r['loss']:.1f} \\\\")
        write(fname, "\\begin{tabular}{llrrrrrrrr}\n\\toprule\n\\textbf{Role} & \\textbf{Family} & \\textbf{Size} & \\textbf{\\%} & $\\bar{d}_{in}$ & $\\bar{d}_{out}$ & $\\overline{\\mathrm{IC}}$ & \\textbf{Anchors} & \\textbf{Density (\\%)} & \\textbf{Loss (\\%)} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- OFAC role profiles
    parts = []
    for key, name in OFAC:
        d = data[key]
        anchor = anchor_role(d)
        lines = []
        for r in role_rows(d):
            star = "$^*$" if r["role"] == anchor else ""
            lines.append(f"R{r['role']}{star} & {FAMILY_TEX[r['family']]} & {fmt_int(r['size'])} & {100*r['share']:.1f} & {r['avg_in']:.2f} & {r['avg_out']:.2f} & {r['seed_count']} & {r['seed_density']:.3f} & {r['connectivity']:.3f} & {fmt_int(r['n_components'])} \\\\")
        parts.append(
            "\\begin{table}[htbp]\n\\centering\n\\caption{ROTOR role profiles for the " + name + f" network ($N$={fmt_int(d['n_nodes'])}, $K$={d['n_roles']}, {sum(r['seed_count'] for r in role_rows(d))} OFAC-listed addresses in graph). "
            "Degrees count individual transfers. Connectivity is the largest-connected-component share of retained addresses after removing every address in the role; the intact network has connectivity 1. $^*$anchor-densest role.}\n"
            f"\\label{{tab:{key.replace('_', '-')}-roles}}\n\\small\n\\begin{{tabular}}{{llrrrrrrrr}}\n\\toprule\n"
            "\\textbf{Role} & \\textbf{Family} & \\textbf{Size} & \\textbf{\\%} & $\\bar{d}_{in}$ & $\\bar{d}_{out}$ & \\textbf{Anchors} & \\textbf{Density (\\%)} & \\textbf{Conn.\\ after} & \\textbf{Comp.\\ after} \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
    write("table_ofac_roles.tex", "\n".join(parts))

    # ---- cross-regime position of the anchor-densest role
    lines = []
    for key, name, _, _ in DATASETS:
        d = data[key]
        rr = role_rows(d)
        anc = next(r for r in rr if r["role"] == anchor_role(d))
        worst = max(rr, key=lambda r: r["loss"])
        lines.append(f"{tex_escape(name)} & R{anc['role']} & {SHORT_FAMILY[anc['family']]} & {100*anc['share']:.1f} & {anc['avg_in']:.1f} & {anc['avg_out']:.1f} & {anc['seed_count']} & {anc['loss']:.1f} & R{worst['role']} & {SHORT_FAMILY[worst['family']]} & {worst['loss']:.1f} \\\\")
    write("table_cross_regime.tex", "\\begin{tabular}{lllrrrrrllr}\n\\toprule\n & \\multicolumn{7}{c}{\\textbf{Anchor-densest role}} & \\multicolumn{3}{c}{\\textbf{Most damaging role}} \\\\\n\\cmidrule(lr){2-8}\\cmidrule(lr){9-11}\n\\textbf{Network} & \\textbf{Role} & \\textbf{Family} & \\textbf{Share (\\%)} & $\\bar{d}_{in}$ & $\\bar{d}_{out}$ & \\textbf{Anchors} & \\textbf{Loss (\\%)} & \\textbf{Role} & \\textbf{Family} & \\textbf{Loss (\\%)} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- dismantling controls
    if controls:
        lines = []
        tagmap = {"israel_tron": "israel", "ukraine_tron": "ukr_tron", "ukraine_eth": "ukr_eth", "ofac_iran_tron": "ofac_iran", "ofac_russia_ukraine_tron": "ofac_ru", "ofac_terrorist_financing_tron": "ofac_tf"}
        for tag, block in controls.items():
            key = tagmap[tag]
            first = True
            for rk, row in block["roles"].items():
                role_loss = 100 * (1 - row["role_connectivity"])
                rand_loss = 100 * (1 - row["random_connectivity_mean"])
                rand_min = 100 * (1 - row["random_connectivity_max"])
                rand_max = 100 * (1 - row["random_connectivity_min"])
                deg_loss = 100 * (1 - row["topdegree_connectivity"])
                label = names[key] if first else ""
                first = False
                lines.append(f"{label} & R{rk.split('_')[1]} & {fmt_int(row['budget'])} & {row['budget_share_pct']:.1f} & {role_loss:.1f} & {rand_loss:.1f} [{rand_min:.1f}, {rand_max:.1f}] & {deg_loss:.1f} \\\\")
            lines.append("\\addlinespace")
        write("table_dismantling_controls.tex", "\\begin{tabular}{llrrrrr}\n\\toprule\n\\textbf{Network} & \\textbf{Role} & \\textbf{Budget} & \\textbf{Share (\\%)} & \\multicolumn{3}{c}{\\textbf{Connectivity loss (\\%)}} \\\\\n\\cmidrule(lr){5-7}\n & & & & \\textbf{role} & \\textbf{random (range, 3 seeds)} & \\textbf{top degree} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- screening extended
    if screening:
        lines = []
        for cname, label in [("all", "all addresses"), ("hop1", "seeds vs.\\ direct counterparties"), ("hop2", "seeds vs.\\ hop-2 addresses")]:
            c = screening["candidate_sets"][cname]
            first = True
            for m, e in c["methods"].items():
                diff = e.get("auc_diff_ic_minus_method")
                dtxt = "--" if diff is None else f"{diff['mean']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]"
                lab = f"{label} ($n$={fmt_int(c['n_candidates'])}, {c['n_positives']} positives)" if first else ""
                first = False
                lines.append(f"{lab} & {m} & {e['auc']:.3f} & [{e['ci_low']:.3f}, {e['ci_high']:.3f}] & {dtxt} \\\\")
            lines.append("\\addlinespace")
        write("table_screening_extended.tex", "\\begin{tabular}{p{4.0cm}lrll}\n\\toprule\n\\textbf{Candidate set} & \\textbf{Score} & \\textbf{AUC} & \\textbf{95\\% CI} & \\textbf{$\\Delta$AUC (IC $-$ score), paired 95\\% CI} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
        ks = list(next(iter(screening["candidate_sets"]["all"]["methods"].values()))["recall_at_k"].keys())
        lines = []
        for m, e in screening["candidate_sets"]["all"]["methods"].items():
            lines.append(f"{m} & " + " & ".join(f"{e['recall_at_k'][k]:.3f}" for k in ks) + " \\\\")
        thr = screening["ic_threshold_flagged"]
        write("table_recall_at_k.tex", "\\begin{tabular}{l" + "r" * len(ks) + "}\n\\toprule\n\\textbf{Score} & " + " & ".join(f"$K$={fmt_int(k)}" for k in ks) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
        lines = [f"{t} & {fmt_int(v['flagged'])} & {v['seeds_recovered']} \\\\" for t, v in thr.items()]
        write("table_ic_thresholds.tex", "\\begin{tabular}{lrr}\n\\toprule\n\\textbf{Threshold} & \\textbf{Addresses flagged} & \\textbf{Sanctioned addresses recovered} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def phenomena_tables() -> None:
    path = ROOT / "phenomena.json"
    if not path.exists():
        return
    ph = json.load(open(path))
    lines = []
    for o in ph["nbctf_orders"]:
        lag = "--" if o["median_days_last_activity_to_signed"] is None else f"{o['median_days_last_activity_to_signed']:.0f}"
        lines.append(f"{o['order']} & {o['signed']} & {o['published'] or '--'} & {o['n_addresses']} & {o['volume_usdt']/1e6:,.1f} & {fmt_int(o['transfers'])} & {lag} \\\\")
    write("table_orders.tex", "\\begin{tabular}{lllrrrr}\n\\toprule\n\\textbf{Order} & \\textbf{Signed} & \\textbf{Published} & \\textbf{Addresses named} & \\textbf{Volume (M USDT)} & \\textbf{Transfers} & \\textbf{Median days, last transfer to signing} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    e, pl = ph["nbctf_event_study"], ph["placebo_event_study"]
    W = ph["window_weeks"]
    rows = []
    for lab, a, b in [("weeks $-26$ to $-1$", 0, W), ("weeks 0 to 4", W, W + 5), ("weeks 5 to 6", W + 5, W + 7), ("weeks 7 to 26", W + 7, 2 * W + 1)]:
        ev = sum(e["volume_usdt"][a:b]) / (b - a); ei = sum(e["inflow_usdt"][a:b]) / (b - a); eo = sum(e["outflow_usdt"][a:b]) / (b - a); ec = sum(e["transfers"][a:b]) / (b - a)
        pv = sum(pl["volume_usdt"][a:b]) / (b - a); pc = sum(pl["transfers"][a:b]) / (b - a)
        rows.append(f"{lab} & {ev/1e6:.1f} & {ei/1e6:.1f} & {eo/1e6:.1f} & {ec:,.0f} & {pv/1e6:.1f} & {pc:,.0f} \\\\")
    write("table_event_study.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{4}{c}{\\textbf{Designated addresses ($n$=" + str(e["n_addresses"]) + ")}} & \\multicolumn{2}{c}{\\textbf{Placebo ($n$=" + f"{pl['n_addresses']:,}" + ")}} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n\\textbf{Window} & \\textbf{Volume/week (M)} & \\textbf{Inflow (M)} & \\textbf{Outflow (M)} & \\textbf{Transfers/week} & \\textbf{Volume/week (M)} & \\textbf{Transfers/week} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    te = ph.get("tether_enforcement", {})
    if te:
        lines = []
        for o in te["by_order"]:
            lag = "--" if o["median_days_signed_to_frozen"] is None else f"{o['median_days_signed_to_frozen']:.0f}"
            dorm = "--" if o.get("median_days_last_transfer_to_freeze") is None else f"{o['median_days_last_transfer_to_freeze']:.0f}"
            lines.append(f"{o['signed']} & {o['n']} & {o['n_frozen']} & {o['n_frozen_before_signing']} & {o.get('n_frozen_within_30d', 0)} & {lag} & {dorm} & {o['lifetime_volume_usdt']/1e6:,.1f} \\\\")
        write("table_enforcement.tex", "\\begin{tabular}{lrrrrrrr}\n\\toprule\n\\textbf{Order signed} & \\textbf{Addresses} & \\textbf{Frozen by Tether} & \\textbf{Frozen before signing} & \\textbf{Frozen within 30 d} & \\textbf{Median days, signing to freeze} & \\textbf{Median days, last transfer to freeze} & \\textbf{Volume (M USDT)} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    u = ph["ukraine"]
    rows = []
    for tag, name in [("tron", "TRON (TRC-20)"), ("eth", "Ethereum (ERC-20)")]:
        r = u[tag]
        w = r["windows"]
        rows.append(f"{name} & {r['first_donation'][:10]} & {fmt_int(r['n_donations'])} & {fmt_int(r['n_donors'])} & {r['volume_usdt']/1e6:.2f} & {w['week1']['volume_usdt']/1e6:.2f} & {w['days0_30']['volume_usdt']/1e6:.2f} & {r['median_donation_usdt']:.0f} & {100*r['share_donations_below_100']:.0f} & {100*r['top1pct_donor_volume_share']:.0f} & {100*r['top10_donor_volume_share']:.0f} \\\\")
    write("table_ukraine.tex", "\\begin{tabular}{llrrrrrrrrr}\n\\toprule\n\\textbf{Chain} & \\textbf{First donation} & \\textbf{Donations} & \\textbf{Donors} & \\textbf{Volume (M)} & \\textbf{Week 1 (M)} & \\textbf{Days 0--30 (M)} & \\textbf{Median (USDT)} & \\textbf{\\% $<$100} & \\textbf{Top 1\\% donors (\\%)} & \\textbf{Top 10 donors (\\%)} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def validation_tables() -> None:
    path = ROOT / "validation_israel_tron.json"
    if not path.exists():
        return
    v = json.load(open(path))
    lines = []
    for name, m in v["methods"].items():
        ac, lv = m.get("anchor_concentration", {}), m.get("label_validity", {})
        lines.append(f"{name} & {ac.get('max_density_pct', float('nan')):.3f} & {100*ac.get('share_in_densest_role', float('nan')):.0f} & {100*ac.get('densest_role_share_of_addresses', float('nan')):.1f} & {ac.get('entropy_normalised', float('nan')):.2f} & {lv.get('nmi', float('nan')):.3f} & {lv.get('purity', float('nan')):.2f} & {m['ari_vs_wcfrm']:.2f} \\\\")
    write("table_role_validation.tex", "\\begin{tabular}{lrrrrrrr}\n\\toprule\n\\textbf{Partition ($K$=" + str(v["k"]) + ")} & \\textbf{Max anchor density (\\%)} & \\textbf{Anchors in densest role (\\%)} & \\textbf{Densest role size (\\%)} & \\textbf{Anchor entropy} & \\textbf{NMI vs labels} & \\textbf{Label purity} & \\textbf{ARI vs ROTOR} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    sil = v["silhouette_over_k"][0]
    ks = sorted(int(k) for k in sil)
    lines = [" & ".join(f"{sil[str(k)]:.3f}" if str(k) in sil else f"{sil[k]:.3f}" for k in ks)]
    write("table_silhouette_k.tex", "\\begin{tabular}{l" + "r" * len(ks) + "}\n\\toprule\n$K$ & " + " & ".join(str(k) for k in ks) + " \\\\\n\\midrule\nsilhouette & " + lines[0] + " \\\\\n\\bottomrule\n\\end{tabular}\n")


def backbone_tables() -> None:
    path = ROOT / "backbone_tests.json"
    if not path.exists():
        return
    bt = json.load(open(path))
    names = {"israel_tron": "NBCTF Israel (TRON)", "ukraine_tron": "Ukraine aid (TRON)", "ukraine_eth": "Ukraine aid (ETH)", "ofac_iran_tron": "OFAC Iran (TRON)", "ofac_russia_ukraine_tron": "OFAC Russia--Ukr. (TRON)", "ofac_terrorist_financing_tron": "OFAC terrorism (TRON)"}
    lines = []
    for tag, r in bt.items():
        a = r.get("A_partition_free")
        if not a:
            continue
        lines.append(f"{names[tag]} & {r['n_anchors']} & {a['anchor_median_degree']:.0f} & {100*a['anchor_share_in_top_1pct_degree']:.0f} & {a['remove_anchors_loss_pct']:.2f} & {a['remove_top_degree_undesignated_loss_pct']:.1f} & {a['remove_random_loss_pct_mean']:.2f} \\\\")
    write("table_backbone_tests.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\n\\textbf{Network} & \\textbf{Anchors $n$} & \\textbf{Anchor median degree} & \\textbf{Anchors in top 1\\% by degree (\\%)} & \\multicolumn{3}{c}{\\textbf{Connectivity loss after removing $n$ addresses (\\%)}} \\\\\n\\cmidrule(lr){5-7}\n & & & & \\textbf{anchors} & \\textbf{top-degree undesignated} & \\textbf{random} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    lines = []
    for tag, r in bt.items():
        b = r["B_raw_feature_partition"]
        same = "same" if b["same_role"] else "different"
        lines.append(f"{names[tag]} & {b['K']} & R{b['anchor_densest_role']} & {b['anchor_densest_density_pct']:.3f} & {b['anchor_densest_loss_pct']:.1f} & R{b['most_damaging_role']} & {b['most_damaging_loss_pct']:.1f} & {same} \\\\")
    write("table_raw_partition.tex", "\\begin{tabular}{lrlrrlrl}\n\\toprule\n\\textbf{Network} & $K$ & \\textbf{Anchor-densest role} & \\textbf{Density (\\%)} & \\textbf{Loss (\\%)} & \\textbf{Most damaging role} & \\textbf{Loss (\\%)} & \\textbf{Roles} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def diffusion_table() -> None:
    path = ROOT / "diffusion_validation.json"
    if not path.exists():
        return
    d = json.load(open(path))
    names = {"israel_tron": "NBCTF Israel (TRON)", "ukraine_tron": "Ukraine aid (TRON)", "ofac_iran_tron": "OFAC Iran (TRON)",
             "ofac_russia_ukraine_tron": "OFAC Russia--Ukr. (TRON)", "ofac_terrorist_financing_tron": "OFAC terrorism (TRON)"}
    order = ["israel_tron", "ukraine_tron", "ofac_terrorist_financing_tron", "ofac_russia_ukraine_tron", "ofac_iran_tron"]
    lines = []
    for tag in order:
        if tag not in d:
            continue
        r = d[tag]
        lines.append(f"{names[tag]} & {fmt_int(r['n_nodes'])} & {r['seconds_linear']:.3f} & {r['seconds_mc_1pass']:.2f} & "
                     f"{r['spearman_mc1_vs_mc1_other_seed']:.2f} & {r['spearman_mc1_vs_mcR']:.2f} & {r['spearman_linear_vs_mcR']:.2f} \\\\")
    write("table_diffusion_validation.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\n\\textbf{Network} & \\textbf{Addresses} & \\textbf{Deterministic (s)} & \\textbf{Monte-Carlo, 1 pass (s)} & \\multicolumn{3}{c}{\\textbf{Spearman $\\rho$}} \\\\\n\\cmidrule(lr){5-7}\n & & & & \\textbf{MC vs MC (other seed)} & \\textbf{MC vs MC average} & \\textbf{deterministic vs MC average} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
    phenomena_tables()
    validation_tables()
    backbone_tables()
    diffusion_table()
