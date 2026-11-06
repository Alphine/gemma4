#!/bin/bash
# usage: run_all.sh <traces.tar.gz> <workdir>   (reads the tar in memory; writes runs.json to workdir)
set -e
cd "$(dirname "$0")"
python build_runs.py "$1" "$2/runs.json"
for s in a1_wall a2_preedit a3_reasoning a4_sinks a5_arms a6_visibility; do
  python $s.py "$2/runs.json" $s.json > $s.txt
done
python slim.py "$2/runs.json" runs_slim.json
