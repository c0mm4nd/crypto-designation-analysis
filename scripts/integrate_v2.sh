#!/usr/bin/env bash
# Pull WCFRM v2 runs from the remote host and regenerate all downstream analyses.
# Usage: bash scripts/integrate_v2.sh [remote]
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${1:-c0mm4nd@172.28.1.3}
mkdir -p v6_runs
rsync -a --exclude '*_model.pt' "$REMOTE:~/wcfrm/repo/v6_runs/" v6_runs/
rsync -a "$REMOTE:~/wcfrm/repo/logs/" logs_v2/ 2>/dev/null || true

run() {  # dataset csv run seeds chain
  python3 scripts/analyze_v2_run.py --csv "$2" --run "$3" --seeds "$4" --chain "$5" --dataset "$1" --out "wcfrm_$1_analysis.json"
}
run israel_tron israel_tron_usdt_edges_2hop.csv v6_runs/v2_israel_tron_seed42 IsraelAddrs.xlsx tron
run ukraine_tron ukraine_tron_usdt_edges_2hop.csv v6_runs/v2_ukraine_tron_seed42 seeds_ukraine_tron.txt tron
python3 scripts/analyze_v2_run.py --agg-edges ukraine_eth_full_edges_agg.csv --agg-nodes ukraine_eth_full_nodes_agg.csv --run v6_runs/v2_ukraine_eth_full_seed42 --seeds seeds_ukraine_eth.txt --chain eth --dataset ukraine_eth --out wcfrm_ukraine_eth_analysis.json
run ofac_iran_tron ofac_iran_tron_usdt_edges_2hop.csv v6_runs/v2_ofac_iran_tron_seed42 ofac_iran.csv tron
run ofac_russia_ukraine_tron ofac_russia_ukraine_tron_usdt_edges_2hop.csv v6_runs/v2_ofac_russia_ukraine_tron_seed42 ofac_russia_ukraine.csv tron
run ofac_terrorist_financing_tron ofac_terrorist_financing_tron_usdt_edges_2hop.csv v6_runs/v2_ofac_terrorist_financing_tron_seed42 ofac_terrorist_financing.csv tron

python3 scripts/compute_dismantling_controls.py
python3 scripts/compute_screening_extended.py
python3 scripts/compute_backbone_tests.py
python3 model/validate_roles.py --csv israel_tron_usdt_edges_2hop.csv --runs 'v6_runs/v2_israel_tron_seed*_report.json' --chain tron --out validation_israel_tron.json
python3 scripts/generate_si_tables.py
python3 scripts/generate_ncomms_figures.py
python3 scripts/generate_source_data.py
echo "integration complete"
