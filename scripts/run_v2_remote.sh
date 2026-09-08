#!/usr/bin/env bash
# Launch WCFRM v2 training for all networks on the remote CPU host.
# Usage (on the remote): bash scripts/run_v2_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/wcfrm/.venv/bin/activate
mkdir -p v2_runs logs
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24

run() {  # tag csv seeds chain K seed threads
  nohup python model/train_v2.py --csv "$2" --tag "$1" --seeds "$3" --chain "$4" --n-clusters "$5" --seed "$6" --threads "$7" \
      --epochs-p1 200 --epochs-p2 100 --cluster-every 10 > "logs/v2_$1_seed$6.log" 2>&1 &
}

# primary network, three training seeds
for s in 42 43 44; do run israel_tron israel_tron_usdt_edges_2hop.csv IsraelAddrs.xlsx tron 9 $s 32; done
# other networks, one seed each (K as in the manuscript)
run ukraine_tron ukraine_tron_usdt_edges_2hop.csv seeds_ukraine_tron.txt tron 3 42 16
run ofac_iran_tron ofac_iran_tron_usdt_edges_2hop.csv ofac_iran.csv tron 9 42 8
run ofac_russia_ukraine_tron ofac_russia_ukraine_tron_usdt_edges_2hop.csv ofac_russia_ukraine.csv tron 9 42 8
run ofac_terrorist_financing_tron ofac_terrorist_financing_tron_usdt_edges_2hop.csv ofac_terrorist_financing.csv tron 9 42 16
wait
echo "all runs finished"
