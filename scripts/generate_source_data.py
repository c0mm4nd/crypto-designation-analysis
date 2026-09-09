#!/usr/bin/env python3
"""Export the numerical source data behind the main-text figures to Source_Data.xlsx.

Nature Communications requires source data for every graph. Sheets are named by
figure panel and are generated from the same result JSONs and helper functions
used by scripts/generate_ncomms_figures.py, so figures and source data cannot drift.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402
OUT = out_path("Source_Data.xlsx")
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
                ("Fig 1b", "Monthly USDT volume through NBCTF-designated addresses."),
                ("Fig 1b orders", "Seizure-order signing dates and the number of addresses each names."),
                ("Fig 1c", "Daily and cumulative USDT donations to the official Aid for Ukraine TRON address; the panel plots the cumulative shares."),
                ("Fig 1d", "Addresses, unique directed edges, roles K and anchor addresses per network."),
                ("Fig 1e", "Share of addresses per learned role in the three focal networks."),
                ("Fig 2a", "Days from last observed transfer to signing, per designated address, with the order that named it."),
                ("Fig 2b-c", "Weekly volume, transfers and share of active addresses relative to the event, designated versus placebo."),
                ("Fig 2d", "Persistence after the order for designated addresses and their counterparties."),
                ("Fig 3a", "Tether blacklisting of designated addresses by seizure order."),
                ("Fig 3b", "Days from signing of the order to Tether blacklisting, per frozen address."),
                ("Fig 3c", "Weekly USDT inflow and outflow relative to the week of blacklisting."),
                ("Fig 3d", "Days from each frozen address's last transfer to its blacklisting."),
                ("Supp Fig 1a", "Per-role mean in- and out-degree and share of addresses, NBCTF TRON network."),
                ("Supp Fig 1b", "Role removal versus budget-matched random and top-degree removal, NBCTF TRON network."),
                ("Fig 3e", "Lifetime USDT inflow against the balance still held at the moment of freezing, per frozen designated address."),
                ("Fig 4a", "Removal test on the complete TRON USDT network: addresses lost from the largest component, throughput of the removed set, and value stranded between surviving addresses."),
                ("Supp Fig 4", "Connectivity loss for the anchor-densest role and the most damaging role in each network (learned partition)."),
                ("Fig 4b", "Connectivity loss from removing the designated addresses and from removing the same number of undesignated hubs, at each crawl boundary and on the complete network."),
                ("Fig 4c", "Cumulative volume share by top share of addresses: counterparties of designated addresses and Ukraine donors."),
                ("Supp Fig 2", "Donation-size histogram, Aid for Ukraine TRON address."),
                ("Supp Fig 3a", "Recall of designated addresses among the K highest-ranked, per score."),
                ("Supp Fig 3b", "ROC-AUC per score over all addresses, with bootstrap intervals."),
                ("Supp Fig 3c", "ROC-AUC per score restricted to designated addresses and their direct counterparties."),
                ("Verified AUC (SI 4.2)", "Naive against verified-negative AUC; reported in Supplementary Note 4.2, not plotted."),
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
        ).to_excel(xl, sheet_name="Supp Fig 3a", index=False)

        # Supplementary screening figure b: AUC with bootstrap CI (IC coverage and centralities from screening_extended.json;
        # betweenness and GNN baselines from wcfrm_results.json)
        rows2b = [{"score": m, "auc": e["auc"], "ci95_low": e["ci_low"], "ci95_high": e["ci_high"], "source": "screening_extended.json"} for m, e in allm.items()]
        rows2b.append({"score": "betweenness", "auc": results["auc"]["Betweenness"]["auc"], "ci95_low": results["auc"]["Betweenness"]["ci_low"], "ci95_high": results["auc"]["Betweenness"]["ci_high"], "source": "wcfrm_results.json"})
        for m in GNN_METHODS:
            rows2b.append({"score": m, "auc": results["auc"][m]["auc"], "ci95_low": results["auc"][m]["ci_low"], "ci95_high": results["auc"][m]["ci_high"], "source": "wcfrm_results.json"})
        pd.DataFrame(rows2b).to_excel(xl, sheet_name="Supp Fig 3b", index=False)

        # Supp Fig 3c: designated addresses against their direct counterparties, the set an
        # analyst reviews, which is the restriction the panel draws.
        hop1 = ext["candidate_sets"]["hop1"]
        pd.DataFrame([{"score": k, "auc": v["auc"], "ci95_low": v["ci_low"], "ci95_high": v["ci_high"],
                       "n_candidates": hop1["n_candidates"], "n_positives": hop1["n_positives"]}
                      for k, v in hop1["methods"].items()]
                     ).to_excel(xl, sheet_name="Supp Fig 3c", index=False)
        # Verified-negative AUC, reported in Supplementary Note 4.2 rather than in a figure
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
        ).to_excel(xl, sheet_name="Verified AUC (SI 4.2)", index=False)

        # Fig 3a-b: per-role statistics for the primary network
        role_frame(names["israel"], data["israel"]).to_excel(xl, sheet_name="Supp Fig 1a", index=False)

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
        pd.DataFrame(rows3c).to_excel(xl, sheet_name="Supp Fig 1b", index=False)

        # Fig 5a: address-level backbone test
        bt = load_json("backbone_tests.json")
        fn = load_json("full_tron_backbone.json")
        fs = load_json("full_tron_stranded.json")["decomposition"]
        dmi = load_json("degree_matched_interval.json")
        iso, thr = dmi["isolated_share_pct_summary"], dmi["throughput_pct_summary"]
        rows4a = [{"removed": "the 400 designated addresses", "addresses_lost_pct": fn["remove_designated_pct"],
                   "throughput_pct": fs["designated"]["incident_pct"], "value_stranded_usdt": fs["designated"]["stranded_usdt"]},
                  {"removed": "400 undesignated, degree-matched (200 draws)", "addresses_lost_pct": iso["mean"],
                   "addresses_lost_ci_low": iso["ci"][0], "addresses_lost_ci_high": iso["ci"][1],
                   "throughput_pct": thr["mean"], "throughput_ci_low": thr["ci"][0], "throughput_ci_high": thr["ci"][1],
                   "value_stranded_usdt": float(sum(dmi["stranded_usdt_exact_draws"]) / len(dmi["stranded_usdt_exact_draws"])),
                   "value_stranded_low": min(dmi["stranded_usdt_exact_draws"]), "value_stranded_high": max(dmi["stranded_usdt_exact_draws"])},
                  {"removed": "400 undesignated, at random", "addresses_lost_pct": fn["remove_random_pct_mean"],
                   "throughput_pct": None, "value_stranded_usdt": None},
                  {"removed": "400 undesignated, highest degree", "addresses_lost_pct": fn["remove_top_degree_undesignated_pct"],
                   "throughput_pct": fs["top_degree_400"]["incident_pct"], "value_stranded_usdt": fs["top_degree_400"]["stranded_usdt"]},
                  {"removed": "1,000 undesignated, highest degree", "addresses_lost_pct": fn["remove_top_degree_undesignated_1000_pct"],
                   "throughput_pct": fs["top_degree_1000"]["incident_pct"], "value_stranded_usdt": fs["top_degree_1000"]["stranded_usdt"]},
                  {"removed": "10,000 undesignated, highest degree", "addresses_lost_pct": fn["remove_top_degree_undesignated_10000_pct"],
                   "throughput_pct": fs["top_degree_10000"]["incident_pct"], "value_stranded_usdt": fs["top_degree_10000"]["stranded_usdt"]}]
        pd.DataFrame(rows4a).to_excel(xl, sheet_name="Fig 4a", index=False)

        bs = load_json("boundary_sensitivity.json")
        rows4b = []
        for tag, name in [("israel_tron", "NBCTF Israel"), ("ofac_iran_tron", "OFAC Iran"),
                          ("ofac_russia_ukraine_tron", "OFAC Russia-Ukraine"),
                          ("ofac_terrorist_financing_tron", "OFAC terrorism")]:
            for key, gl in (("two_hop", "two hops"), ("hop1", "one hop")):
                rows4b.append({"network": name, "boundary": gl,
                               "loss_removing_designated_pct": bs[tag][key]["remove_designated_pct"],
                               "loss_removing_top_degree_undesignated_pct": bs[tag][key]["remove_top_degree_undesignated_pct"]})
        rows4b.append({"network": "complete TRON USDT network", "boundary": "none",
                       "loss_removing_designated_pct": fn["remove_designated_pct"],
                       "loss_removing_top_degree_undesignated_pct": fn["remove_top_degree_undesignated_pct"]})
        pd.DataFrame(rows4b).to_excel(xl, sheet_name="Fig 4b", index=False)

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
        pd.DataFrame(rows).to_excel(xl, sheet_name="Supp Fig 4", index=False)

        # phenomenon panels
        ph = load_json("phenomena.json")
        m = ph["nbctf_monthly"]
        pd.DataFrame({"month": m["month"], "designated_volume_usdt": m["designated_volume_usdt"], "designated_active_addresses": m["designated_active_addresses"], "network_volume_usdt": m["network_volume_usdt"]}).to_excel(xl, sheet_name="Fig 1b", index=False)
        pd.DataFrame(ph["nbctf_orders"]).to_excel(xl, sheet_name="Fig 1b orders", index=False)
        u = ph["ukraine"]["tron"]["daily"]
        _d = pd.to_datetime(u["date"]); _el = (_d - pd.Timestamp("2022-02-24")).days + 1
        _v = np.array(u["volume_usdt"], dtype=float); _n = np.array(u["donations"], dtype=float)
        pd.DataFrame({"date": u["date"], "days_since_invasion": _el, "volume_usdt": u["volume_usdt"],
                      "donations": u["donations"], "donors": u["donors"],
                      "cumulative_volume_share_pct": _v.cumsum() / _v.sum() * 100,
                      "cumulative_donation_share_pct": _n.cumsum() / _n.sum() * 100}).to_excel(xl, sheet_name="Fig 1c", index=False)
        t = ph["nbctf_timing"]
        pd.DataFrame([{"order": o, "signed": t["signed_by_order"][o], "days_last_transfer_to_signing": v}
                      for o, vs in t["days_last_activity_to_signed_by_order"].items() for v in vs]
                     ).to_excel(xl, sheet_name="Fig 2a", index=False)
        e, pl = ph["nbctf_event_study"], ph["placebo_event_study"]
        pd.DataFrame({"week_relative_to_event": e["weeks"], "designated_volume_usdt": e["volume_usdt"], "designated_inflow_usdt": e["inflow_usdt"], "designated_outflow_usdt": e["outflow_usdt"],
                      "designated_transfers": e["transfers"], "designated_share_active": e["per_address"]["share_active_by_week"],
                      "placebo_volume_usdt": pl["volume_usdt"], "placebo_transfers": pl["transfers"], "placebo_share_active": pl["per_address"]["share_active_by_week"]}).to_excel(xl, sheet_name="Fig 2b-c", index=False)
        c = ph["counterparty_persistence"]
        pd.DataFrame([{"group": "designated addresses", "n": e["n_addresses"], "share_active_after_order": t["share_active_after_signing"], "volume_share_after_order": t["volume_after_signing_share"]},
                      {"group": "undesignated counterparties", "n": c["n_counterparties"], "share_active_after_order": c["share_active_after_order"], "share_active_90d_after_order": c["share_active_90d_after_order"], "volume_share_after_order": c["volume_share_after_order"]}]).to_excel(xl, sheet_name="Fig 2d", index=False)
        k = ph["concentration"]
        xs = [i * 100 / 199 for i in range(200)]
        pd.DataFrame({"counterparty_rank": k["counterparty_lorenz_log_rank"],
                      "counterparty_top_share_pct": [r / k["n_counterparties_ranked"] * 100 for r in k["counterparty_lorenz_log_rank"]],
                      "counterparties_cumulative_volume_share": k["counterparty_lorenz_log"]}).to_excel(xl, sheet_name="Fig 4c", index=False)
        u_ = ph["ukraine"]["tron"]
        pd.DataFrame({"donor_rank": u_["donor_lorenz_log_rank"],
                      "donor_top_share_pct": [r / u_["n_donors_ranked"] * 100 for r in u_["donor_lorenz_log_rank"]],
                      "donors_cumulative_volume_share": u_["donor_lorenz_log"]}).to_excel(xl, sheet_name="Fig 4c donors", index=False)
        te = ph["tether_enforcement"]
        pd.DataFrame(te["by_order"]).to_excel(xl, sheet_name="Fig 3a", index=False)
        pd.DataFrame({"days_signing_to_blacklist": te["days_signed_to_frozen"]}).to_excel(xl, sheet_name="Fig 3b", index=False)
        ef = te["event_study_freeze"]
        pd.DataFrame({"week_relative_to_freeze": ef["weeks"], "inflow_usdt": ef["inflow_usdt"], "outflow_usdt": ef["outflow_usdt"], "transfers": ef["transfers"]}).to_excel(xl, sheet_name="Fig 3c", index=False)
        pd.DataFrame({"days_last_transfer_to_blacklist": te["days_last_transfer_to_freeze"]}).to_excel(xl, sheet_name="Fig 3d", index=False)
        pd.DataFrame({"lifetime_inflow_usdt": te["per_address_lifetime_inflow_usdt"],
                      "balance_at_freeze_usdt": te["per_address_balance_at_freeze_usdt"]}).to_excel(xl, sheet_name="Fig 3e", index=False)
        h = ph["ukraine"]["tron"]["donation_size_hist"]
        pd.DataFrame({"bin_lower_usdt": h["bin_edges_usdt"][:-1], "bin_upper_usdt": h["bin_edges_usdt"][1:], "donations": h["counts"]}).to_excel(xl, sheet_name="Supp Fig 2", index=False)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
