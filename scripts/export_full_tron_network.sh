#!/usr/bin/env bash
# Export the complete TRON USDT transfer network from an archival node.
#
# The two-hop neighbourhoods used elsewhere in this study are crawled outward from the
# designated addresses, and no structural quantity measured in them is identified: the
# removal test reverses sign between the one-hop and two-hop boundaries. This script builds
# the network that has no boundary — every USDT TRC-20 Transfer event on TRON before the
# cut, aggregated to unique directed address pairs.
#
# Addresses are carried as cityHash64 of the 20-byte address rather than as text, which
# keeps the edge list to 20 bytes per pair (about 15 GB) instead of about 70. At 2.1e8
# addresses the expected number of colliding pairs is n^2/2^65 ~ 1e-3, and the chance that
# a collision involves one of the few hundred designated addresses is of order 1e-8.
#
# The export is split into 32 buckets by the hash of the sender so that each GROUP BY fits
# in memory. Each bucket is retried until its file is a whole number of 20-byte records,
# because a query cancelled under memory pressure leaves a truncated file that parses as
# garbage rather than failing.
#
# Usage: bash scripts/export_full_tron_network.sh [output_dir] [cut_ms]
set -uo pipefail
OUT=${1:-tron_full2}
CUT=${2:-1735689600000}          # 2025-01-01T00:00:00Z in milliseconds
USDT=a614f803b6fd780986a42c78ec9c7f77e6ded13c          # TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t, without the 0x41 prefix
TRANSFER=ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef

CH=$(docker ps -qf name=clickhouse-analyticaldb | head -1)
mkdir -p "$OUT" logs

for B in $(seq 0 31); do
  F=$OUT/bucket_$B.bin
  for ATTEMPT in 1 2 3 4; do
    if [ -f "$F.ok" ]; then break; fi
    # assumeNotNull matters: topic1 and topic2 are Nullable, so cityHash64 of a substring of
    # them is Nullable(UInt64) and RowBinary writes a null flag byte before each value,
    # making the record 22 bytes rather than 20.
    docker exec -i "$CH" clickhouse-client --user "${CH_USER:-w3r}" --password "${CH_PASSWORD:?set CH_PASSWORD}" \
      --max_bytes_before_external_group_by=6000000000 --max_memory_usage=12000000000 \
      --format RowBinary --query "
SELECT cityHash64(assumeNotNull(substring(topic1,25,40))) AS fi,
       cityHash64(assumeNotNull(substring(topic2,25,40))) AS ti,
       toUInt32(count()) AS cnt
FROM tron.events
WHERE address='$USDT' AND topic0='$TRANSFER'
  AND topic2 IS NOT NULL AND blockTimestamp < $CUT
  AND cityHash64(assumeNotNull(substring(topic1,25,40))) % 32 = $B
GROUP BY fi, ti" > "$F" 2>> logs/export.err
    rc=$?; sz=$(stat -c %s "$F")
    if [ $rc -eq 0 ] && [ "$sz" -gt 0 ] && [ $(( sz % 20 )) -eq 0 ]; then touch "$F.ok"; fi
    echo "bucket $B attempt $ATTEMPT rc=$rc size=$sz mod=$(( sz % 20 ))"
  done
done
echo "buckets complete: $(ls "$OUT"/*.ok 2>/dev/null | wc -l)/32"

# ---------------------------------------------------------------------------------------
# The value-carrying export. Every quantity in the paper that is weighted by USDT rather than
# by address count reads this one: the throughput and stranded-value decomposition, the
# degree-matched throughput interval, and the count of zero-value pairs. It is the same
# aggregation as above with the transferred amount summed per directed pair, so the records
# are 24 bytes (two UInt64 hashes and a Float64) rather than 20.
#
# The amount is the data field of the Transfer event, a 64-character big-endian hex string,
# scaled by 1e6 for USDT's six decimals. Only the low 8 bytes can hold a real amount, so the
# last 16 hex characters are decoded and reversed for reinterpretAsUInt64, which reads
# little-endian. This reproduces the deposited per-address histories exactly; amounts above
# 1e8 USDT are integer-overflow artefacts and are dropped downstream, as for the crawls.
OUTV=${3:-tron_full_val}
mkdir -p "$OUTV"
for B in $(seq 0 31); do
  F=$OUTV/bucket_$B.bin
  for ATTEMPT in 1 2 3 4; do
    if [ -f "$F.ok" ]; then break; fi
    docker exec -i "$CH" clickhouse-client --user "${CH_USER:-w3r}" --password "${CH_PASSWORD:?set CH_PASSWORD}" \
      --max_bytes_before_external_group_by=6000000000 --max_memory_usage=12000000000 \
      --format RowBinary --query "
SELECT cityHash64(assumeNotNull(substring(topic1,25,40))) AS fi,
       cityHash64(assumeNotNull(substring(topic2,25,40))) AS ti,
       sum(reinterpretAsUInt64(reverse(unhex(substring(assumeNotNull(data), 49, 16)))) / 1000000.) AS val
FROM tron.events
WHERE address='$USDT' AND topic0='$TRANSFER'
  AND topic2 IS NOT NULL AND blockTimestamp < $CUT
  AND cityHash64(assumeNotNull(substring(topic1,25,40))) % 32 = $B
GROUP BY fi, ti" > "$F" 2>> logs/export_val.err
    rc=$?; sz=$(stat -c %s "$F")
    if [ $rc -eq 0 ] && [ "$sz" -gt 0 ] && [ $(( sz % 24 )) -eq 0 ]; then touch "$F.ok"; fi
    echo "value bucket $B attempt $ATTEMPT rc=$rc size=$sz mod=$(( sz % 24 ))"
  done
done
echo "value buckets complete: $(ls "$OUTV"/*.ok 2>/dev/null | wc -l)/32"

# Row and address counts quoted in the paper, from the same filter.
docker exec -i "$CH" clickhouse-client --user "${CH_USER:-w3r}" --password "${CH_PASSWORD:?}" --query "
SELECT count() AS transfers_before_cut,
       uniqExact(assumeNotNull(substring(topic1,25,40))) AS senders
FROM tron.events
WHERE address='$USDT' AND topic0='$TRANSFER' AND topic2 IS NOT NULL AND blockTimestamp < $CUT"
