#!/usr/bin/env bash
# Exact number of distinct USDT Transfer events on TRON before the cut, against the number of
# stored rows. The archive node's event table holds duplicate rows where blocks were ingested
# more than once, so the two differ; events are distinct on (transactionHash, logIndex). The
# count is summed over 64 hash buckets of the transaction hash so that each uniqExact fits in
# memory. Writes distinct_transfers.json beside the other analysis outputs.
#
# Usage: CH_URL=http://host:8123 CH_AUTH=user:password scripts/count_distinct_transfers.sh
set -euo pipefail
: "${CH_URL:?set CH_URL}" "${CH_AUTH:?set CH_AUTH}"
USDT=a614f803b6fd780986a42c78ec9c7f77e6ded13c; TR=ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
total=0; rows=0
for B in $(seq 0 63); do
  r=$(curl -s --max-time 3000 -u "$CH_AUTH" "$CH_URL/?max_memory_usage=12000000000" --data-binary "SELECT count(), uniqExact(transactionHash, logIndex) FROM tron.events WHERE address='$USDT' AND topic0='$TR' AND topic2 IS NOT NULL AND blockTimestamp < 1735689600000 AND cityHash64(transactionHash) % 64 = $B")
  set -- $r; rows=$((rows + $1)); total=$((total + $2)); echo "bucket $B rows $1 distinct $2 | cumulative rows $rows distinct $total"
done
echo "TOTAL rows $rows distinct_transfers $total"
out=$(dirname "$0")/../distinct_transfers.json
printf '{\n "description": "Exact number of distinct USDT transfer events on TRON before 1 January 2025 in the archive node event table, against the number of stored rows; rows are distinct on (transactionHash, logIndex) and the table holds duplicate rows where blocks were ingested more than once. Counted with uniqExact over 64 buckets by cityHash64(transactionHash) %% 64 (scripts/count_distinct_transfers.sh).",\n "rows": %d,\n "distinct_transfers": %d,\n "duplicate_rows": %d,\n "duplicate_share_pct": %s\n}\n' "$rows" "$total" "$((rows - total))" "$(python3 -c "print(100*($rows-$total)/$rows)")" > "$out"
echo "saved $out"
