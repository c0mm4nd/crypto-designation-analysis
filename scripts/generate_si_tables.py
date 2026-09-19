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
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402
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
    out_path(name).write_text(body)
    print(f"wrote {name}")


def main() -> None:
    data = {key: load_json(fname) for key, _, fname, _ in DATASETS}
    names = {key: tex_escape(name) for key, name, _, _ in DATASETS}
    controls = json.load(open(find("dismantling_controls.json"))) if find("dismantling_controls.json", required=False).exists() else None
    screening = json.load(open(find("screening_extended.json"))) if find("screening_extended.json", required=False).exists() else None

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
    write("table_backbone_tests.tex", "\\begin{tabular}{lrrrrrr}\n\\toprule\n\\textbf{Network} & \\textbf{Anchors $n$} & \\textbf{Anchor median deg.} & \\textbf{Anchors in top 1\\% (\\%)} & \\multicolumn{3}{c}{\\textbf{Connectivity loss (\\%)}} \\\\\n\\cmidrule(lr){5-7}\n & & & & \\textbf{anchors} & \\textbf{top-degree undes.} & \\textbf{random} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    lines = []
    for tag, r in bt.items():
        b = r["B_raw_feature_partition"]
        same = "same" if b["same_role"] else "different"
        lines.append(f"{names[tag]} & {b['K']} & R{b['anchor_densest_role']} & {b['anchor_densest_density_pct']:.3f} & {b['anchor_densest_loss_pct']:.1f} & R{b['most_damaging_role']} & {b['most_damaging_loss_pct']:.1f} & {same} \\\\")
    write("table_raw_partition.tex", "\\begin{tabular}{lrlrrlrl}\n\\toprule\n\\textbf{Network} & $K$ & \\textbf{Anchor-densest role} & \\textbf{Density (\\%)} & \\textbf{Loss (\\%)} & \\textbf{Most damaging role} & \\textbf{Loss (\\%)} & \\textbf{Roles} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def zero_value_table() -> None:
    path = ROOT / "zero_value_sensitivity.json"
    if not path.exists():
        return
    zv = json.load(open(path))
    a, b = zv["all_pairs"], zv["value_carrying_pairs"]
    labels = {"designated": "Remove all 400 designated", "top_degree_undesignated": "Remove 400 highest-degree undesignated",
              "degree_matched_undesignated": "Remove 400 degree-matched undesignated (one draw)", "random_undesignated": "Remove 400 random undesignated (one draw)"}
    def usdt(v):
        return f"{v/1e9:.2f} billion" if v >= 1e9 else f"{v:,.0f} USDT"
    lines = [f"Pairs & {a['n_pairs']:,} & {b['n_pairs']:,} \\\\",
             f"Addresses with an edge & {a['n_addresses_with_an_edge']:,} & {b['n_addresses_with_an_edge']:,} \\\\",
             f"Largest component before removal (\\% of addresses) & {100*a['baseline_lcc_share']:.4f} & {100*b['baseline_lcc_share']:.4f} \\\\",
             "\\midrule"]
    for k, lab in labels.items():
        ra, rb = a["removals"][k], b["removals"][k]
        lines.append(f"{lab}: addresses lost (\\%) & {ra['connectivity_loss_pct']:.4f} & {rb['connectivity_loss_pct']:.4f} \\\\")
        lines.append(f"\\quad throughput (\\% of value) & {ra['incident_pct']:.4f} & {rb['incident_pct']:.4f} \\\\")
        lines.append(f"\\quad value stranded between survivors & {usdt(ra['stranded_usdt'])} & {usdt(rb['stranded_usdt'])} \\\\")
    write("table_zero_value.tex",
          "\\caption{\\textbf{The removal test with and without the pairs that carry no value.} The complete TRON USDT network to 1 January 2025 is built with no value filter, so a zero-value transfer creates an edge. The test of Supplementary Table~\\ref{tab:full-network} is repeated on the subgraph of pairs carrying a positive amount; removal sets are chosen on each graph's own degrees, and the share of addresses lost is taken over the addresses that have an edge in the graph in question. Dropping the zero-value pairs widens the gap between the designated set and the highest-degree undesignated addresses on both component-based measures; throughput is unaffected by construction (\\texttt{scripts/zero\\_value\\_sensitivity.py}).}\n\\label{tab:zero-value}\n"
          "\\begin{tabular}{lrr}\n\\toprule\n & \\textbf{All pairs} & \\textbf{Value-carrying pairs} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def custody_tables() -> None:
    if not find("custody.json", required=False).exists():
        return
    cu = json.load(open(find("custody.json")))
    def row(name, a):
        fz = a["n_frozen"]
        left = f"{a['left_between_signing_and_freeze_total']/1e6:.1f}" if a["left_between_signing_and_freeze_total"] is not None else "--"
        atf = f"{a['balance_at_freeze_total']/1e6:.2f}" if a["balance_at_freeze_total"] is not None else "--"
        return (f"{name} & {a['n']} & {a['deposit_by_own_label']} & {a['deposit_by_recipient']} & {a['deposit_by_behaviour']} & {a['deposit_any']} & "
                f"{a['median_recipients']:.0f} & {a['median_top_recipient_share']:.2f} & {a['median_sweep_share_24h']:.2f} & {a['median_dwell_hours']:.0f} & "
                f"{a['median_peak_balance']/1e3:.1f} & {100*a['peak_balance_total']/a['inflow_total']:.1f} & {100*a['share_balance_at_signing_gt_1']:.0f} & "
                f"{a['balance_at_signing_total']/1e6:.2f} & {fz} & {atf} & {left} \\\\")
    orders = sorted(cu["by_order"].items(), key=lambda kv: kv[1]["n"], reverse=True)
    lines = [row(tex_escape(k), v) for k, v in orders if v["n"] >= 3]
    lines.append("\\midrule")
    lines.append(row("All orders", cu["overall"]))
    write("table_custody.tex",
          "\\caption{\\textbf{Custody indicators for the designated addresses, by order.} From the complete USDT histories of the "
          f"{cu['n_addresses_with_transfer']} designated addresses with a transfer (to {cu['history_end']}). An address is counted as an exchange deposit "
          "address on three kinds of evidence: its own third-party label names it as an exchange user address (label); at least 90\\% of its outflow "
          "value went to one recipient that carries an exchange label (recipient); or at least 90\\% of its inflow value left within 24 hours, value "
          "dwelt in it for a median of at most 24 hours and it forwarded to at most three recipients (behaviour). Labels cover few addresses, so the first "
          "two columns are lower bounds. Recipients, top-recipient share, sweep share (share of inflow value gone within 24 hours), dwell (hours until "
          "an inflow is forwarded) and peak balance are per-address medians; peak/inflow is the summed peak balance over the summed inflow; "
          "balance at signing is the share of addresses holding more than 1 USDT and the total held; freeze columns give the total held when Tether "
          "blacklisted the address and the total that left the addresses between signing and freezing. Orders with fewer than three addresses are "
          "pooled into the last row only (\\texttt{scripts/compute\\_custody.py}).}\n\\label{tab:custody}\n"
          "\\begin{tabular}{lrrrrrrrrrrrrrrrr}\n\\toprule\n"
          " & & \\multicolumn{4}{c}{\\textbf{Deposit-address evidence}} & \\multicolumn{4}{c}{\\textbf{Forwarding (medians)}} & \\multicolumn{2}{c}{\\textbf{Peak balance}} & "
          "\\multicolumn{2}{c}{\\textbf{At signing}} & \\multicolumn{3}{c}{\\textbf{Freeze}} \\\\\n"
          "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}\\cmidrule(lr){15-17}\n"
          "\\textbf{Order} & $n$ & label & recipient & behaviour & any & recip. & top share & sweep & dwell (h) & median (k USDT) & /inflow (\\%) & "
          "$>1$ USDT (\\%) & total (M) & $n$ & held (M) & left after signing (M) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    if find("publication_alignment.json", required=False).exists():
        pa = json.load(open(find("publication_alignment.json")))
        lines = []
        for k, v in pa["orders"].items():
            if "ratio_weeks_7_26" not in v:
                continue
            pw = v["publication_week"]
            between = f"{v['ratio_weeks_between_signing_and_publication']:.2f}" if v.get("ratio_weeks_between_signing_and_publication") is not None else "--"
            after = f"{v['ratio_weeks_after_publication_to_26']:.4f}" if v.get("ratio_weeks_after_publication_to_26") is not None else "--"
            first = v["first_week_below_10pct_and_staying"]
            lines.append(f"{tex_escape(k)} & {v['signed']} & {v['published']} & {v['days_signing_to_publication']} & {v['n_addresses']} & "
                         f"{v['pre_mean_weekly_volume']/1e6:.2f} & {between} & {after} & {v['ratio_weeks_7_26']:.4f} & {first if first is not None else '--'} \\\\")
        write("table_publication.tex",
              "\\caption{\\textbf{Flows through each order's addresses between signing and publication.} Weekly USDT volume through the addresses of "
              "each order inside the event window, relative to the mean of the 26 weeks before signing: in the weeks from signing up to the week of "
              "publication, in the weeks after publication to week 26, and in weeks 7 to 26 (the event-study window). The last column is the first week "
              "after signing from which volume stays below 10\\% of the pre-signing mean. For the two orders whose addresses were still moving money "
              "at signing, the fall coincides with publication rather than signature (\\texttt{scripts/compute\\_publication\\_alignment.py}).}\n"
              "\\label{tab:publication}\n\\begin{tabular}{lllrrrrrrr}\n\\toprule\n"
              "\\textbf{Order} & \\textbf{Signed} & \\textbf{Published} & \\textbf{Days} & $n$ & \\textbf{Pre (M/week)} & "
              "\\textbf{Signing to publication} & \\textbf{After publication} & \\textbf{Weeks 7--26} & \\textbf{Below 10\\% from week} \\\\\n\\midrule\n"
              + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    if find("counterparty_full.json", required=False).exists():
        cf = json.load(open(find("counterparty_full.json")))
        ph = json.load(open(find("phenomena.json")))["counterparty_persistence"]
        def r(name, a, med=True):
            m = f"{100*a['median_per_address_post_share']:.0f}" if med and "median_per_address_post_share" in a else "--"
            return f"{name} & {a['n']:,} & {100*a['share_active_after_order']:.0f} & {100*a['share_active_90d_after_order']:.0f} & {100*a['volume_share_after_order']:.0f} & {m} \\\\"
        lines = [r("Designated addresses (complete histories)", cf["designated"]),
                 r("Counterparties with pre-order contact (complete histories)", cf["counterparties"]),
                 r("\\quad excluding the 1\\% largest by volume", cf["counterparties_excluding_top_1pct_by_volume"]),
                 r("Counterparties reached by the crawl (crawled histories, earlier basis)", dict(ph, n=ph["n_counterparties"]), med=False)]
        chk = cf["check_designated_transfer_count_csv_vs_node"]
        write("table_counterparty_full.tex",
              "\\caption{\\textbf{Counterparty persistence on complete histories.} Every undesignated address that transacted with one of the 178 "
              "in-window designated addresses before that address's order, dated by the earliest such order, with its full USDT history read from the "
              f"archive node to {cf['data_end']} and the designated addresses measured identically. Share active: any transfer after the order; 90 d: a transfer more "
              "than 90 days after it; volume share: the pooled group's USDT volume after the order over its volume in the window; median: the per-address "
              "share of volume after the order. The pooled volume share is carried by exchange-scale counterparties, so it is also given without the 1\\% "
              f"largest by volume. The last row is the earlier measure on crawled histories. The node and the per-address export agree on the designated addresses' transfer counts to within {100*abs(chk['node']-chk['csv'])/chk['csv']:.1f}\\% "
              "(\\texttt{scripts/compute\\_counterparty\\_full.py}).}\n\\label{tab:counterparty-full}\n"
              "\\begin{tabular}{lrrrrr}\n\\toprule\n\\textbf{Group} & $n$ & \\textbf{Active after (\\%)} & \\textbf{Active 90 d (\\%)} & \\textbf{Volume share after (\\%)} & \\textbf{Median (\\%)} \\\\\n\\midrule\n"
              + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def post_signing_table() -> None:
    if not find("post_signing_flows.json", required=False).exists():
        return
    ps = json.load(open(find("post_signing_flows.json")))
    a, o = ps["all_frozen"], ps.get("ASO 29/23")
    cats = ["unlabelled", "designated (same programme)", "labelled exchange or payment processor", "other labelled"]
    names = {"unlabelled": "Unlabelled addresses", "designated (same programme)": "Other designated addresses",
             "labelled exchange or payment processor": "Labelled exchanges and payment processors", "other labelled": "Other labelled addresses"}
    def col(x, k):
        return f"{100*x['share_by_category'].get(k, 0):.2f}"
    lines = [f"USDT sent (million) & {a['total_usdt']/1e6:.1f} & {o['total_usdt']/1e6:.1f} \\\\",
             f"Sending addresses & {a['n_sending_addresses']} & {o['n_sending_addresses']} \\\\",
             f"Distinct recipients & {a['n_recipients']:,} & {o['n_recipients']:,} \\\\", "\\midrule"]
    lines += [f"{names[k]} (\\% of value) & {col(a, k)} & {col(o, k)} \\\\" for k in cats]
    lines += ["\\midrule", f"Largest recipient (\\% of value) & {100*a['share_top1']:.1f} & {100*o['share_top1']:.1f} \\\\",
              f"Ten largest recipients (\\% of value) & {100*a['share_top10']:.1f} & {100*o['share_top10']:.1f} \\\\"]
    top = ps["top20_recipients_all_frozen"][:10]
    top_lines = [f"{r['rank']} & {100*r['share_of_outflow']:.1f} & {r['first_seen']} & {r['transfers']:,} & {r['counterparties']:,} & {'yes' if r['in_top400_by_activity'] else 'no'} \\\\" for r in top]
    write("table_post_signing.tex",
          "\\caption{\\textbf{Where the USDT that left frozen designated addresses between signing and freeze went.} Outflows of the frozen designated "
          "addresses between the signing of their order and Tether's blacklisting, from complete histories, by category of recipient; the "
          "May 2023 order (ASO 29/23) is shown separately because it supplies almost all of the value. Recipients are categorised by third-party "
          "entity labels, which are sparse, so the labelled shares are lower bounds. The lower panel describes the ten largest recipients of the "
          "pooled outflow from their own complete histories: the date of their first USDT transfer, their lifetime transfer count, their distinct "
          "counterparties, and whether they are among the 400 most active addresses of the complete network "
          "(\\texttt{scripts/compute\\_post\\_signing\\_flows.py}).}\n\\label{tab:post-signing}\n"
          "\\begin{tabular}{lrr}\n\\toprule\n & \\textbf{All frozen addresses} & \\textbf{ASO 29/23} \\\\\n\\midrule\n" + "\n".join(lines) +
          "\n\\bottomrule\n\\end{tabular}\n\n\\medskip\n\n\\begin{tabular}{lrrrrl}\n\\toprule\n\\textbf{Recipient rank} & \\textbf{Share (\\%)} & "
          "\\textbf{First transfer} & \\textbf{Transfers} & \\textbf{Counterparties} & \\textbf{Top-400 hub} \\\\\n\\midrule\n" + "\n".join(top_lines) +
          "\n\\bottomrule\n\\end{tabular}\n")


def ofac_timing_table() -> None:
    if not find("ofac_timing.json", required=False).exists():
        return
    ot = json.load(open(find("ofac_timing.json")))
    prog = {"ofac_iran": "Iran", "ofac_russia_ukraine": "Russia", "ofac_terrorist_financing": "Terrorism"}
    def f(x, fmt="{:.0f}"):
        return "--" if x is None else fmt.format(x)
    def row(name, v, listed="", entity_listed="", programmes=""):
        return (f"{name} & {programmes} & {listed} & {entity_listed} & {v['n_listed']} & {f(v['median_days_last_to_listing'])} & "
                f"{f(100*v['share_dormant_90d'] if v['share_dormant_90d'] is not None else None)} & {f(100*v['share_active_after_listing'])} & "
                f"{f(v['median_days_listing_to_freeze'])} & {v['n_frozen_before_listing']} & {v['balance_at_freeze_total']/1e3:.1f} & "
                f"{v['peak_balance_total']/1e6:.2f} & {v['volume_total']/1e6:.1f} \\\\")
    ents = sorted(ot["by_entity"].items(), key=lambda kv: kv[1]["listed"])
    short = {"OBSHCHESTVO S OGRANICHENNOI OTVETSTVENNOSTYU KONSTRUKTORSKOE BYURO VOSTOK": "KB Vostok (OOO)", "AL-LAW Tawfiq Muhammad Sa'id": "Al-Law, Tawfiq Muhammad Sa'id",
             "Al-Jamal Sa'id Ahmad Muhammad": "Al-Jamal, Sa'id Ahmad Muhammad", "Gambashidze Ilya Andreevich": "Gambashidze, Ilya Andreevich", "Chirkinyan Elena": "Chirkinyan, Elena", "Shafiu Ali": "Shafiu, Ali"}
    lines = [row(tex_escape(short.get(e, e)), v, v["listed"], v["entity_listed"] if v["entity_listed"] != v["listed"] else "--",
                 ", ".join(prog[p] for p in v["programmes"])) for e, v in ents]
    lines.append("\\midrule")
    lines += [row(f"All {prog[p]}", v, programmes="") for p, v in ot["by_programme"].items()]
    lines.append(row("All OFAC TRON USDT addresses", ot["pooled"]))
    write("table_ofac_timing.tex",
          "\\caption{\\textbf{Timing and Tether enforcement for the OFAC-listed TRON USDT addresses.} Every TRON USDT address listed under the three OFAC programmes "
          "used in this study, grouped by the listed entity, with the same measures as for the NBCTF orders computed from complete histories to 1 July 2026, the end of the node export "
          f"(last transfer {ot['history_end']}), and the blacklist to 1 January 2026. Listed: the date of the SDN List publication that added the address ({ot['n_dated_by_change_history']} of {ot['pooled']['n_listed']} "
          "addresses are dated by the list's change history; the remaining address predates it and carries its entity's date). Entity: the date the entity itself "
          "was first listed, where earlier. Lag: median days from the address's last transfer before listing to the listing. Dormant: share inactive for more "
          "than 90 days at listing. Active after: share with any transfer after listing. Freeze: median days from listing to Tether's blacklisting, and the "
          "number blacklisted before listing. Balance at freeze, summed peak balance and lifetime volume are in thousand, million and million USDT "
          "(\\texttt{scripts/compute\\_ofac\\_timing.py}, \\texttt{scripts/fetch\\_ofac\\_address\\_dates.py}).}\n\\label{tab:ofac-timing}\n"
          "\\begin{tabular}{llllrrrrrrrrr}\n\\toprule\n\\textbf{Entity} & \\textbf{Programme} & \\textbf{Listed} & \\textbf{Entity} & $n$ & \\textbf{Lag (d)} & "
          "\\textbf{Dormant (\\%)} & \\textbf{Active after (\\%)} & \\textbf{Freeze (d)} & \\textbf{Before} & \\textbf{At freeze (k)} & \\textbf{Peak (M)} & \\textbf{Volume (M)} \\\\\n\\midrule\n"
          + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


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
    zero_value_table()
    custody_tables()
    post_signing_table()
    ofac_timing_table()
