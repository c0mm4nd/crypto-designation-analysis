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

The complete TRON USDT network (2.23 billion distinct transfers) is not deposited: it is
40 GB in the binary bucket form the analysis uses. `scripts/export_full_tron_network.sh` rebuilds it
from any archival TRON node indexed in ClickHouse, and `scripts/rerun_complete_network_local.py`
streams the same deduplicated graph straight into the `graph_cache_*.npy` arrays the analyses read.

## analysis/ additions in v1.1.2

`overflow_records.json` records, per crawled network, how many records were dropped as
integer-overflow artefacts of the crawl; `scripts/count_overflow_records.py` reproduces it.
`zero_value_pairs.json`, `wcfrm_results.json` and `verified_metrics.json` are read by
`scripts/generate_source_data.py` and by the Methods, and are now deposited with the rest so
that the figure source data can be regenerated from the archive alone.

## Running from the archive (v1.2.0)

`scripts/paths.py` resolves every input against both layouts, so the scripts run unchanged
whether they sit in the working tree (outputs in the root, record-level inputs in `ch_data/`)
or in this archive (outputs in `analysis/`, inputs in `data/`).

`data/` now also carries the seizure-order list with the signing and publication date of each
of the twenty orders (`IsraelAddrs.xlsx`), the OFAC and Aid for Ukraine anchor lists, and the
crawled two-hop edge lists of the five TRON networks. Two inputs remain outside the deposit:
the complete TRON USDT network, which `scripts/export_full_tron_network.sh` rebuilds from any
archival node — the script now emits both the count-only and the value-carrying export — and
the third-party entity labels, which are used under their providers' terms. The scripts run
without the labels and report the label-derived statistics, which the manuscript gives as
lower bounds, as empty.

`scripts/label_complete_network_hubs.py` names the highest-activity undesignated addresses of
the complete network, and `analysis/complete_network_hubs.json` records the ranking and the
aggregate entity composition (per-address labels are not redistributed).

## Rebuilding the complete network: what the node must provide

`scripts/export_full_tron_network.sh` reads a ClickHouse table `tron.events` holding decoded
TRON contract event logs with at least these columns: `address` (contract, 40 hex chars without
the 0x41 prefix), `topic0`, `topic1`, `topic2` (64-hex-character strings; `topic2` nullable),
`data` (64-hex-character string), and `blockTimestamp` (milliseconds). The USDT contract is
`a614f803b6fd780986a42c78ec9c7f77e6ded13c` and the Transfer topic
`ddf252ad...b3ef`. The script runs `clickhouse-client` inside a Docker container named
`clickhouse-analyticaldb*`; set `CH_USER`/`CH_PASSWORD`, or edit the `docker exec` line to
point at your client. Each of the 32 buckets needs about 12 GB of memory and the two exports
take roughly a day on a single node; the count-only export is ~15 GB and the value-carrying
one ~18 GB. `scripts/designated_hashes.py` writes `designated_hash.tsv` next to the analysis
outputs, which is where `full_network_*.py` read it.

## Duplicate event rows in the source table (re-run completed in v1.6.0)

`tron.events` holds duplicate rows where blocks were ingested more than once: over the USDT
Transfer events before 1 January 2025 there are 2,375,557,775 rows but 2,225,933,042
distinct (transactionHash, logIndex) pairs, 6.30% fewer (`analysis/distinct_transfers.json`,
counted exactly by `scripts/count_distinct_transfers.sh`). The per-address export always
deduplicated; the complete-network export first deposited (v1.5.1 and earlier) did not, so its
value sums were overstated by about seven per cent. In v1.6.0 the complete network was rebuilt
with the duplicates removed (`scripts/rerun_complete_network_local.py`) and every complete-network
analysis was re-run on it. Duplicate rows never created pairs, so the address set (213,338,784),
the pair set (744,543,633) and every connectivity result are byte-identical to the earlier build
(`full_tron_backbone.json`, `full_tron_kcore.json`, `zero_value_pairs.json` unchanged); the
value-carrying outputs changed: total value 15.50 -> 14.43 trillion USDT, value incident to the
designated addresses 10.65 -> 9.94 billion USDT (now equal to the per-address histories), stranded
value for the 400 / 1,000 / 10,000 highest-degree undesignated addresses 3.41 / 5.04 / 9.61 ->
3.15 / 4.70 / 8.98 billion USDT, throughput 44.95 / 47.06 / 54.28 -> 44.35 / 46.48 / 53.81%. The
designated stranded value (590 USDT) and stranded share (0.022%) are unchanged.
`scripts/compare_dedup_outputs.py` prints the full old-versus-new comparison.
`scripts/degree_matched_interval.py` gained `--phase draws|exact`, which splits its two memory
peaks so it runs on a machine with less free memory; the two phases merge into one output that
is identical to a single-process run.

## Zero-value sensitivity (v1.6.1)

The complete network is built with no value filter, so a zero-value transfer creates an edge.
`scripts/zero_value_sensitivity.py` repeats the removal test on the subgraph of pairs carrying a
positive amount (695,762,180 of 744,543,633 pairs; 199,343,561 of 213,338,784 addresses keep an
edge) and writes `analysis/zero_value_sensitivity.json`. Dropping the zero-value pairs widens the
gap: removing the designated addresses costs 0.0005% of addresses instead of 0.0023% and strands
the same 590 USDT, while removing the 400 highest-degree undesignated addresses costs 28.90%
instead of 26.87% and strands 3.39 instead of 3.15 billion USDT. The same run supplies the
throughput and stranded value of the random-removal control (one draw), previously reported only
as an address count.
