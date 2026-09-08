# Cryptocurrency designation: analysis code and outputs

Code and analysis outputs for *Cryptocurrency designation reaches the addresses that moved the
money, not the infrastructure that carries it*.

The paper asks what designation reaches. It uses three kinds of data: the complete on-chain
histories of the addresses named in Israeli counter-terrorism seizure orders and in three OFAC
programmes; two-hop USDT transfer neighbourhoods crawled around those addresses; and the complete
USDT transfer network on TRON, exported from an archival node with no seeding and no sampling.

The distinction matters, because a structural quantity measured in a crawled neighbourhood is not
identified: `scripts/compute_boundary_sensitivity.py` shows that the answer to whether designated
addresses hold their network together reverses sign between the one-hop and the two-hop boundary,
in every sanctions network. Every structural claim in the paper is therefore made on the complete
network.

## Layout

```
scripts/     analysis and figure generation
model/       the role model (encoder, prototype head, training)
labels/      third-party entity label handling, used only for validation
analysis/    the JSON outputs every number in the paper is drawn from, and the
             ClickHouse queries that produced the raw exports
tables/      the LaTeX tables generated from those outputs
```

## Reproducing

The transfer data are public blockchain records. Two sources are needed: a TRON archival node
indexed into ClickHouse (databases `tron`, `ethereum`), and the seizure-order address lists
published by the issuing authorities.

```
export CH_PASSWORD=...                       # ClickHouse credentials
bash scripts/export_full_tron_network.sh     # the complete network, 32 buckets
python scripts/designated_hashes.py          # designated addresses to network identifiers
python scripts/full_network_backbone.py      # removal test, address counts
python scripts/full_network_value_removal.py # the same, weighted by value
python scripts/full_network_stranded.py      # incident versus stranded decomposition
python scripts/degree_matched_interval.py    # 200-draw degree-matched control
python scripts/compute_phenomena.py          # timing, enforcement, event study, concentration
python scripts/compute_event_robustness.py   # event-date, pre-trend, cluster and placebo checks
python scripts/compute_uncertainty.py        # bootstrap intervals under two units
python scripts/compute_boundary_sensitivity.py
python scripts/generate_si_tables.py
python scripts/generate_ncomms_figures.py
```

The complete-network scripts read paths configured at the top of each file; they were run on a
single machine with 125 GB of memory and no GPU. The intermediate exports are large (the complete
network is 15 GB as 744,543,633 aggregated directed pairs) and are not included here; the queries
that produce them are in `analysis/`.

## What the outputs contain

| File | Contents |
|---|---|
| `full_tron_backbone.json` | removal test on the complete network, address counts |
| `full_tron_value_removal.json` | the same, weighted by transferred value |
| `full_tron_stranded.json` | value split into the removed set's own throughput and value stranded between survivors |
| `full_tron_kcore.json` | what the component loss is made of: degree-one share, neighbourhood isolation |
| `degree_matched_interval.json` | 200 degree-matched draws, both measures |
| `boundary_sensitivity.json` | the same removals at the one-hop and two-hop boundaries |
| `phenomena.json` | designation timing, on-chain enforcement, event study, counterparty persistence, concentration, donations |
| `event_robustness.json` | alternative event dates, pre-trend, leave-one-order-out, cluster bootstrap, volume-matched placebo |
| `uncertainty.json` | bootstrap intervals under address and order resampling |
| `score_convergence.json` | spectral radius and Katz comparison for the importance score |

## Note on what is not here

Address-level role assignments and importance scores for undesignated addresses are not released.
They would attach a machine-generated label to individuals who have not been named by any
authority, and the paper's own screening analysis shows that structural position separates
designated addresses from their counterparties only weakly. The designated addresses are already
public.

## Licence

Code is released under the MIT licence. Analysis outputs are released under CC BY 4.0.

## data/ (added in v1.1.0)

The record-level inputs the Data Availability statement names, so that the analysis outputs
in `analysis/` can be recomputed rather than only re-read:

- `designated_usdt_transfers_complete_490.csv` — complete USDT TRC-20 transfer history of all
  490 designated addresses, exported from the contract event logs of an archival TRON node.
  Every timing, event-study and enforcement quantity in the paper is computed from this file.
- `designated_tether_enforcement.csv`, `designated_blacklist_match.csv`,
  `tron_usdt_blacklist_added.csv`, `tron_usdt_blackfunds_destroyed.csv` — the Tether blacklist
  and fund-destruction events emitted by the USDT contract, and the match to designated addresses.
- `ukraine_eth_anchor_usdt.csv` — complete USDT history of the Aid for Ukraine Ethereum address.
- `*.sql` — the ClickHouse queries that produced the exports.

The complete TRON USDT network (2.38 billion transfers) is not deposited: it is 40 GB in the
binary bucket form the analysis uses. `scripts/export_full_tron_network.sh` rebuilds it from any
archival TRON node indexed in ClickHouse.

## analysis/ additions in v1.1.2

`overflow_records.json` records, per crawled network, how many records were dropped as
integer-overflow artefacts of the crawl; `scripts/count_overflow_records.py` reproduces it.
`zero_value_pairs.json`, `wcfrm_results.json` and `verified_metrics.json` are read by
`scripts/generate_source_data.py` and by the Methods, and are now deposited with the rest so
that the figure source data can be regenerated from the archive alone.
