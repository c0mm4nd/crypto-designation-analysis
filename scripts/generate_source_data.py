#!/usr/bin/env python3
"""Export the numerical source data behind the main-text figures to Source_Data.xlsx.

Nature Communications requires source data for every graph. Sheets are named by
figure panel and are generated from the same result JSONs and helper functions
used by scripts/generate_ncomms_figures.py, so figures and source data cannot drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Source_Data.xlsx"
sys.path.insert(0, str(ROOT / "scripts"))

from generate_ncomms_figures import DATASETS, anchor_role, load_json, role_rows  # noqa: E402

GNN_METHODS = ["GCN+KMeans", "GATv2+KMeans", "GNN-FiLM+KMeans", "GAT+KMeans", "GIN+KMeans", "SuperGAT+KMeans", "GraphSAGE+KMeans"]
VERIFIED_TAGS = [
    "israel_tron",
    "ofac_terrorist_financing_tron",
    "ofac_terrorist_financing_eth",
    "ofac_russia_ukraine_eth",
    "ofac_russia_ukraine_tron",
    "ofac_iran_tron",
]


def role_frame(name: str, data: dict) -> pd.DataFrame:
    rows = []
    for r in role_rows(data):
        rows.append(
            {
                "network": name,
                "role": f"R{r['role']}",
                "functional_family": r["family"],
                "n_addresses": r["size"],
                "share_pct": round(100 * r["share"], 4),
                "mean_in_degree": r["avg_in"],
                "mean_out_degree": r["avg_out"],
                "anchor_count": r["seed_count"],
                "anchor_density_pct": round(r["seed_density"], 5),
                "post_removal_connectivity": r["connectivity"],
                "connectivity_loss_pct": round(r["loss"], 4),
                "post_removal_components": r["n_components"],
                "post_removal_path_efficiency": r["path_efficiency"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = {key: load_json(fname) for key, _, fname, _ in DATASETS}
    names = {key: name for key, name, _, _ in DATASETS}
    results = load_json("wcfrm_results.json")
    verified = {r["tag"]: r for r in load_json("verified_metrics.json")}

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        readme = pd.DataFrame(
            [
                ("Fig 1a", "Study-design schematic; no plotted data."),
                ("Fig 1b", "Monthly USDT volume through NBCTF-designated addresses and seizure-order signing dates."),
                ("Fig 1c", "Daily USDT donations to the official Aid for Ukraine TRON address."),
                ("Fig 1d", "Addresses, unique directed edges, roles K and anchor addresses per network."),
                ("Fig 1e", "Share of addresses per learned role in the three focal networks."),
                ("Fig 2a", "Days from last observed transfer to signing of the seizure order, per designated address."),
                ("Fig 2b-c", "Weekly volume, transfers and share of active addresses relative to the event, designated versus placebo."),
                ("Fig 2d", "Persistence after the order for designated addresses and their counterparties."),
                ("Fig 3a-b", "Tether blacklisting of designated addresses by order; days from signing to freeze."),
                ("Fig 3c-d", "Weekly in/outflow relative to the freeze; days from last transfer to freeze."),
                ("Supp Fig 2a-b", "Per-role statistics for the NBCTF TRON network."),
                ("Supp Fig 2c", "Role removal versus budget-matched random and top-degree removal, NBCTF TRON network."),
                ("Fig 4a", "Removal test on the complete TRON USDT network: addresses lost from the largest component and throughput of the removed set."),
                ("SuppFig cross-network", "Connectivity loss for the anchor-densest role and the most damaging role in each network (learned partition)."),
                ("Fig 4b", "Cumulative volume share by top share of addresses: counterparties of designated addresses and Ukraine donors."),
                ("Fig 4c", "Donation-size histogram, Aid for Ukraine TRON address."),
                ("Supp Fig screening", "Recall@K and ROC-AUC per score on the primary benchmark; naive versus verified-negative AUC."),
            ],
            columns=["sheet", "description"],
        )
        readme.to_excel(xl, sheet_name="README", index=False)

        # Fig 1b
        scale_rows = []
        for key, name, _, setting in DATASETS:
            d = data[key]
            scale_rows.append(
                {
                    "network": name,
                    "setting": setting,
                    "n_addresses": d["n_nodes"],
                    "n_transfers_edges": d.get("n_edges"),
                    "n_roles_K": d["n_roles"],
                    "n_anchor_addresses": sum(r["seed_count"] for r in role_rows(d)),
                }
            )
        pd.DataFrame(scale_rows).to_excel(xl, sheet_name="Fig 1d", index=False)

        # Fig 1c
        comp = []
        for key in ("israel", "ukr_tron", "ukr_eth"):
            df = role_frame(names[key], data[key])[["network", "role", "functional_family", "n_addresses", "share_pct", "anchor_count"]]
            df["anchor_densest_role"] = df["role"] == f"R{anchor_role(data[key])}"
            comp.append(df)
        pd.concat(comp, ignore_index=True).to_excel(xl, sheet_name="Fig 1e", index=False)

        # Fig 2a: recall@K for IC coverage and centralities
        ext = load_json("screening_extended.json")
        allm = ext["candidate_sets"]["all"]["methods"]
        ks = list(allm["diffusion reach (ROTOR)"]["recall_at_k"].keys())
        pd.DataFrame(
            [{"score": m, **{f"recall_at_{k}": e["recall_at_k"][k] for k in ks}} for m, e in allm.items()]
        ).to_excel(xl, sheet_name="SuppFig screening a", index=False)

        # Supplementary screening figure b: AUC with bootstrap CI (IC coverage and centralities from screening_extended.json;
        # betweenness and GNN baselines from wcfrm_results.json)
        rows2b = [{"score": m, "auc": e["auc"], "ci95_low": e["ci_low"], "ci95_high": e["ci_high"], "source": "screening_extended.json"} for m, e in allm.items()]
        rows2b.append({"score": "betweenness", "auc": results["auc"]["Betweenness"]["auc"], "ci95_low": results["auc"]["Betweenness"]["ci_low"], "ci95_high": results["auc"]["Betweenness"]["ci_high"], "source": "wcfrm_results.json"})
        for m in GNN_METHODS:
            rows2b.append({"score": m, "auc": results["auc"][m]["auc"], "ci95_low": results["auc"][m]["ci_low"], "ci95_high": results["auc"][m]["ci_high"], "source": "wcfrm_results.json"})
        pd.DataFrame(rows2b).to_excel(xl, sheet_name="SuppFig screening b", index=False)

        # Fig 2c
        pd.DataFrame(
            [
                {
                    "dataset": tag,
                    "n_addresses": verified[tag]["n_nodes"],
                    "n_positives": verified[tag]["n_positives"],
                    "n_verified_negatives": verified[tag]["n_verified_neg"],
                    "naive_auc": verified[tag]["naive_auc"],
                    "naive_ci95_low": verified[tag]["naive_auc_ci"][0],
                    "naive_ci95_high": verified[tag]["naive_auc_ci"][1],
                    "verified_auc": verified[tag]["verified_auc"],
                    "verified_ci95_low": verified[tag]["verified_auc_ci"][0],
                    "verified_ci95_high": verified[tag]["verified_auc_ci"][1],
                }
                for tag in VERIFIED_TAGS
            ]
        ).to_excel(xl, sheet_name="SuppFig screening c", index=False)

        # Fig 3a-b: per-role statistics for the primary network
        role_frame(names["israel"], data["israel"]).to_excel(xl, sheet_name="Supp Fig 2a-b", index=False)

        # Fig 3c: role removal versus budget-matched controls (primary network)
        ctrl = load_json("dismantling_controls.json")["israel_tron"]["roles"]
        rows3c = []
        for rk_, c in ctrl.items():
            rows3c.append(
                {
                    "role": f"R{rk_.split('_')[1]}",
                    "budget_addresses_removed": c["budget"],
                    "budget_share_pct": c["budget_share_pct"],
                    "role_removal_connectivity_loss_pct": round(100 * (1 - c["role_connectivity"]), 4),
                    "random_removal_loss_pct_mean": round(100 * (1 - c["random_connectivity_mean"]), 4),
                    "random_removal_loss_pct_min": round(100 * (1 - c["random_connectivity_max"]), 4),
                    "random_removal_loss_pct_max": round(100 * (1 - c["random_connectivity_min"]), 4),
                    "top_degree_removal_loss_pct": round(100 * (1 - c["topdegree_connectivity"]), 4),
                }
            )
        pd.DataFrame(rows3c).to_excel(xl, sheet_name="Supp Fig 2c", index=False)

        # Fig 5a: address-level backbone test
        bt = load_json("backbone_tests.json")
        fn = load_json("full_tron_backbone.json"); fv = load_json("full_tron_value_removal.json")["removals"]
        fs = load_json("full_tron_stranded.json")["decomposition"]
        rows4a = [{"removed": "the 400 designated addresses", "addresses_lost_pct": fn["remove_designated_pct"],
                   "throughput_pct": fv["designated"]["value"], "value_stranded_usdt": fs["designated"]["stranded_usdt"]},
                  {"removed": "400 undesignated, degree-matched", "addresses_lost_pct": fn["remove_degree_matched_pct_mean"],
                   "throughput_pct": fv["degree_matched_undesignated"]["value"], "value_stranded_usdt": None},
                  {"removed": "400 undesignated, at random", "addresses_lost_pct": fn["remove_random_pct_mean"],
                   "throughput_pct": None, "value_stranded_usdt": None},
                  {"removed": "400 undesignated, highest degree", "addresses_lost_pct": fn["remove_top_degree_undesignated_pct"],
                   "throughput_pct": fv["top_degree_400"]["value"], "value_stranded_usdt": fs["top_degree_400"]["stranded_usdt"]},
                  {"removed": "1,000 undesignated, highest degree", "addresses_lost_pct": fn["remove_top_degree_undesignated_1000_pct"],
                   "throughput_pct": fv["top_degree_1000"]["value"], "value_stranded_usdt": fs["top_degree_1000"]["stranded_usdt"]},
                  {"removed": "10,000 undesignated, highest degree", "addresses_lost_pct": fn["remove_top_degree_undesignated_10000_pct"],
                   "throughput_pct": fv["top_degree_10000"]["value"], "value_stranded_usdt": fs["top_degree_10000"]["stranded_usdt"]}]
        pd.DataFrame(rows4a).to_excel(xl, sheet_name="Fig 4a", index=False)

        # Supplementary cross-network role figure
        rows = []
        for key, name, _, _ in DATASETS:
            rr = role_rows(data[key])
            anc = next(r for r in rr if r["role"] == anchor_role(data[key]))
            worst = max(rr, key=lambda r: r["loss"])
            rows.append(
                {
                    "network": name,
                    "anchor_densest_role": f"R{anc['role']}",
                    "anchor_densest_role_connectivity_loss_pct": round(anc["loss"], 4),
                    "most_damaging_role": f"R{worst['role']}",
                    "most_damaging_role_connectivity_loss_pct": round(worst["loss"], 4),
                }
            )
        pd.DataFrame(rows).to_excel(xl, sheet_name="SuppFig cross-network", index=False)

        # phenomenon panels
        ph = load_json("phenomena.json")
        m = ph["nbctf_monthly"]
        pd.DataFrame({"month": m["month"], "designated_volume_usdt": m["designated_volume_usdt"], "designated_active_addresses": m["designated_active_addresses"], "network_volume_usdt": m["network_volume_usdt"]}).to_excel(xl, sheet_name="Fig 1b", index=False)
        pd.DataFrame(ph["nbctf_orders"]).to_excel(xl, sheet_name="Fig 1b orders", index=False)
        u = ph["ukraine"]["tron"]["daily"]
        pd.DataFrame({"date": u["date"], "volume_usdt": u["volume_usdt"], "donations": u["donations"], "donors": u["donors"]}).to_excel(xl, sheet_name="Fig 1c", index=False)
        t = ph["nbctf_timing"]
        pd.DataFrame({"days_last_transfer_to_signing": t["days_last_activity_to_signed"]}).to_excel(xl, sheet_name="Fig 2a", index=False)
        e, pl = ph["nbctf_event_study"], ph["placebo_event_study"]
        pd.DataFrame({"week_relative_to_event": e["weeks"], "designated_volume_usdt": e["volume_usdt"], "designated_inflow_usdt": e["inflow_usdt"], "designated_outflow_usdt": e["outflow_usdt"],
                      "designated_transfers": e["transfers"], "designated_share_active": e["per_address"]["share_active_by_week"],
                      "placebo_volume_usdt": pl["volume_usdt"], "placebo_transfers": pl["transfers"], "placebo_share_active": pl["per_address"]["share_active_by_week"]}).to_excel(xl, sheet_name="Fig 2b-c", index=False)
        c = ph["counterparty_persistence"]
        pd.DataFrame([{"group": "designated addresses", "n": e["n_addresses"], "share_active_after_order": t["share_active_after_signing"], "volume_share_after_order": t["volume_after_signing_share"]},
                      {"group": "undesignated counterparties", "n": c["n_counterparties"], "share_active_after_order": c["share_active_after_order"], "share_active_90d_after_order": c["share_active_90d_after_order"], "volume_share_after_order": c["volume_share_after_order"]}]).to_excel(xl, sheet_name="Fig 2d", index=False)
        k = ph["concentration"]
        xs = [i * 100 / 199 for i in range(200)]
        pd.DataFrame({"top_share_of_addresses_pct": xs, "counterparties_cumulative_volume_share": k["counterparty_lorenz"], "ukraine_donors_cumulative_volume_share": ph["ukraine"]["tron"]["donor_lorenz"]}).to_excel(xl, sheet_name="Fig 4b", index=False)
        te = ph["tether_enforcement"]
        pd.DataFrame(te["by_order"]).to_excel(xl, sheet_name="Fig 3a", index=False)
        pd.DataFrame({"days_signing_to_blacklist": te["days_signed_to_frozen"]}).to_excel(xl, sheet_name="Fig 3b", index=False)
        ef = te["event_study_freeze"]
        pd.DataFrame({"week_relative_to_freeze": ef["weeks"], "inflow_usdt": ef["inflow_usdt"], "outflow_usdt": ef["outflow_usdt"], "transfers": ef["transfers"]}).to_excel(xl, sheet_name="Fig 3c", index=False)
        pd.DataFrame({"days_last_transfer_to_blacklist": te["days_last_transfer_to_freeze"]}).to_excel(xl, sheet_name="Fig 3d", index=False)
        h = ph["ukraine"]["tron"]["donation_size_hist"]
        pd.DataFrame({"bin_lower_usdt": h["bin_edges_usdt"][:-1], "bin_upper_usdt": h["bin_edges_usdt"][1:], "donations": h["counts"]}).to_excel(xl, sheet_name="Fig 4c", index=False)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
