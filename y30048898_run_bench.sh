#!/bin/bash


BASE=/mnt/xds/sfs/l00972766
ROOT=$BASE/ds41_mini
# SPEC=""
# SPEC="$SPEC 1k@0.0/b1/r1 1k@0.1/b1/r1 1k@0.6/b1/r1 1k@0.7/b1/r1 1k@0.8/b1/r1 1k@0.9/b1/r1 1k@0.95/b1/r1 1k@0.96/b1/r1 1k@0.97/b1/r1 1k@0.98/b1/r1 1k@0.99/b1/r1"
# SPEC="$SPEC 2k@0.0/b1/r1 2k@0.1/b1/r1 2k@0.6/b1/r1 2k@0.7/b1/r1 2k@0.8/b1/r1 2k@0.9/b1/r1 2k@0.95/b1/r1 2k@0.96/b1/r1 2k@0.97/b1/r1 2k@0.98/b1/r1 2k@0.99/b1/r1"
# SPEC="$SPEC 4k@0.0/b1/r1 4k@0.1/b1/r1 4k@0.6/b1/r1 4k@0.7/b1/r1 4k@0.8/b1/r1 4k@0.9/b1/r1 4k@0.95/b1/r1 4k@0.96/b1/r1 4k@0.97/b1/r1 4k@0.98/b1/r1 4k@0.99/b1/r1"
# SPEC="$SPEC 8k@0.0/b1/r1 8k@0.1/b1/r1 8k@0.6/b1/r1 8k@0.7/b1/r1 8k@0.8/b1/r1 8k@0.9/b1/r1 8k@0.95/b1/r1 8k@0.96/b1/r1 8k@0.97/b1/r1 8k@0.98/b1/r1 8k@0.99/b1/r1"
# SPEC="$SPEC 12k@0.0/b1/r1 12k@0.1/b1/r1 12k@0.6/b1/r1 12k@0.7/b1/r1 12k@0.8/b1/r1 12k@0.9/b1/r1 12k@0.95/b1/r1 12k@0.96/b1/r1 12k@0.97/b1/r1 12k@0.98/b1/r1 12k@0.99/b1/r1"
# SPEC="$SPEC 16k@0.0/b1/r1 16k@0.1/b1/r1 16k@0.6/b1/r1 16k@0.7/b1/r1 16k@0.8/b1/r1 16k@0.9/b1/r1 16k@0.95/b1/r1 16k@0.96/b1/r1 16k@0.97/b1/r1 16k@0.98/b1/r1 16k@0.99/b1/r1"
# SPEC="$SPEC 20k@0.0/b1/r1 20k@0.1/b1/r1 20k@0.6/b1/r1 20k@0.7/b1/r1 20k@0.8/b1/r1 20k@0.9/b1/r1 20k@0.95/b1/r1 20k@0.96/b1/r1 20k@0.97/b1/r1 20k@0.98/b1/r1 20k@0.99/b1/r1"
# SPEC="$SPEC 24k@0.0/b1/r1 24k@0.1/b1/r1 24k@0.6/b1/r1 24k@0.7/b1/r1 24k@0.8/b1/r1 24k@0.9/b1/r1 24k@0.95/b1/r1 24k@0.96/b1/r1 24k@0.97/b1/r1 24k@0.98/b1/r1 24k@0.99/b1/r1"
# SPEC="$SPEC 32k@0.0/b1/r1 32k@0.1/b1/r1 32k@0.6/b1/r1 32k@0.7/b1/r1 32k@0.8/b1/r1 32k@0.9/b1/r1 32k@0.95/b1/r1 32k@0.96/b1/r1 32k@0.97/b1/r1 32k@0.98/b1/r1 32k@0.99/b1/r1"
# SPEC="$SPEC 40k@0.0/b1/r1 40k@0.1/b1/r1 40k@0.6/b1/r1 40k@0.7/b1/r1 40k@0.8/b1/r1 40k@0.9/b1/r1 40k@0.95/b1/r1 40k@0.96/b1/r1 40k@0.97/b1/r1 40k@0.98/b1/r1 40k@0.99/b1/r1"
# SPEC="$SPEC 48k@0.0/b1/r1 48k@0.1/b1/r1 48k@0.6/b1/r1 48k@0.7/b1/r1 48k@0.8/b1/r1 48k@0.9/b1/r1 48k@0.95/b1/r1 48k@0.96/b1/r1 48k@0.97/b1/r1 48k@0.98/b1/r1 48k@0.99/b1/r1"
# SPEC="$SPEC 56k@0.0/b1/r1 56k@0.1/b1/r1 56k@0.6/b1/r1 56k@0.7/b1/r1 56k@0.8/b1/r1 56k@0.9/b1/r1 56k@0.95/b1/r1 56k@0.96/b1/r1 56k@0.97/b1/r1 56k@0.98/b1/r1 56k@0.99/b1/r1"
# SPEC="$SPEC 64k@0.0/b1/r1 64k@0.1/b1/r1 64k@0.6/b1/r1 64k@0.7/b1/r1 64k@0.8/b1/r1 64k@0.9/b1/r1 64k@0.95/b1/r1 64k@0.96/b1/r1 64k@0.97/b1/r1 64k@0.98/b1/r1 64k@0.99/b1/r1"
# SPEC="$SPEC 80k@0.0/b1/r1 80k@0.1/b1/r1 80k@0.6/b1/r1 80k@0.7/b1/r1 80k@0.8/b1/r1 80k@0.9/b1/r1 80k@0.95/b1/r1 80k@0.96/b1/r1 80k@0.97/b1/r1 80k@0.98/b1/r1 80k@0.99/b1/r1"
# SPEC="$SPEC 96k@0.0/b1/r1 96k@0.1/b1/r1 96k@0.6/b1/r1 96k@0.7/b1/r1 96k@0.8/b1/r1 96k@0.9/b1/r1 96k@0.95/b1/r1 96k@0.96/b1/r1 96k@0.97/b1/r1 96k@0.98/b1/r1 96k@0.99/b1/r1"
# SPEC="$SPEC 112k@0.0/b1/r1 112k@0.1/b1/r1 112k@0.6/b1/r1 112k@0.7/b1/r1 112k@0.8/b1/r1 112k@0.9/b1/r1 112k@0.95/b1/r1 112k@0.96/b1/r1 112k@0.97/b1/r1 112k@0.98/b1/r1 112k@0.99/b1/r1"
# SPEC="$SPEC 128k@0.0/b1/r1 128k@0.1/b1/r1 128k@0.6/b1/r1 128k@0.7/b1/r1 128k@0.8/b1/r1 128k@0.9/b1/r1 128k@0.95/b1/r1 128k@0.96/b1/r1 128k@0.97/b1/r1 128k@0.98/b1/r1 128k@0.99/b1/r1"
# SPEC="$SPEC 256k@0.0/b1/r1 256k@0.1/b1/r1 256k@0.6/b1/r1 256k@0.7/b1/r1 256k@0.8/b1/r1 256k@0.9/b1/r1 256k@0.95/b1/r1 256k@0.96/b1/r1 256k@0.97/b1/r1 256k@0.98/b1/r1 256k@0.99/b1/r1"
# SPEC="$SPEC 512k@0.0/b1/r1 512k@0.1/b1/r1 512k@0.6/b1/r1 512k@0.7/b1/r1 512k@0.8/b1/r1 512k@0.9/b1/r1 512k@0.95/b1/r1 512k@0.96/b1/r1 512k@0.97/b1/r1 512k@0.98/b1/r1 512k@0.99/b1/r1"
# SPEC="$SPEC 1M@0.0/b1/r1 1M@0.1/b1/r1 1M@0.6/b1/r1 1M@0.7/b1/r1 1M@0.8/b1/r1 1M@0.9/b1/r1 1M@0.95/b1/r1 1M@0.96/b1/r1 1M@0.97/b1/r1 1M@0.98/b1/r1 1M@0.99/b1/r1"

# echo $BASE
# echo $ROOT
# echo $SPEC

BWLIMIT=/mnt/xds/sfs/y30048898/bandwidth/bwlimit
DEVICES=0,1,2,3,4,5,6,7
TS=$(date +%Y%m%d_%H%M)

for BANDWIDTH in 1T 1.6T 2T 4T; do
  echo "============== [$(date '+%F %T')] Setting BANDWIDTH=$BANDWIDTH =============="

  # 先 restore 再按新带宽 apply，避免残留旧限速
  cd /mnt/xds/sfs/y30048898/bandwidth
  $BWLIMIT restore --devices $DEVICES && \
  $BWLIMIT apply --all --limit "$BANDWIDTH"
  $BWLIMIT status --devices $DEVICES
  echo "---------------------------------------------------------------------"

  OUT=$BASE/tmp/1M_high_hit/${BANDWIDTH}
  cd "$ROOT"
  bash tools/run_cases.sh $OUT

  echo -e "---------------------------------------------------------------------\n\n\n\n\n\n\n\n\n\n\n\n"

done

cd /mnt/xds/sfs/y30048898/bandwidth
$BWLIMIT restore --devices $DEVICES