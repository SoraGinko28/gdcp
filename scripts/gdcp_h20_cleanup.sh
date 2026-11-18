#!/usr/bin/env bash
# Run on H20 after every GDCP test process has completed.
set -euo pipefail
export TZ=America/Los_Angeles
date '+%Y-%m-%d %H:%M:%S PST'
docker rm -f iobench-gdcp-v1
rm -rf -- /data00/iobench/gdcp-test-checkpoints
test ! -e /data00/iobench/gdcp-test-checkpoints
if docker ps -a --format '{{.Names}}' | grep -q '^iobench-gdcp-'; then
  echo 'ERROR: a GDCP container remains'
  exit 1
fi
echo 'GDCP cleanup PASS: no GDCP containers or test checkpoints remain'
date '+%Y-%m-%d %H:%M:%S PST'
