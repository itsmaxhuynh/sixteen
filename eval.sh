#!/bin/bash
# Compare checkpoints and search depths on the same seeded games.
#
# usage: ./eval.sh [-n games] [-s seed] [-t threads] [-p "1 2 3"] [-x "play args"] ckpt [ckpt ...]
#   -n games    games per run (default 1000)
#   -s seed     game i uses environment seed seed+i (default 1), the same for every run
#   -t threads  worker threads (default 1). Use 1 when the move times matter: threads share
#               memory bandwidth, so move times measured with more threads run slow
#   -p plies    expectimax plies to try (default "1 2")
#   -x args     extra --play args for every run, e.g. -x "downgrade=32768" or -x "time=50"
#
# Prints one row per (checkpoint, ply): mean score with 95% interval, reach rates of 8192 /
# 16384 / 32768 with 95% intervals, and the average / max time per move.
set -euo pipefail
cd "$(dirname "$0")"

games=1000 seed=1 threads=1 plies="1 2" extra=""
while getopts "n:s:t:p:x:" opt; do
  case $opt in
    n) games=$OPTARG ;;
    s) seed=$OPTARG ;;
    t) threads=$OPTARG ;;
    p) plies=$OPTARG ;;
    x) extra=$OPTARG ;;
    *) sed -n '2,14p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { sed -n '2,14p' "$0"; exit 1; }

make -s sixteen
printf "%-28s %3s %18s %22s %22s %22s %9s %9s %6s %7s\n" checkpoint ply "mean ± ci95" "8192 %" "16384 %" "32768 %" "move_us" "max_ms" "over" "avg_ply"
for ckpt in "$@"; do
  for ply in $plies; do
    line=$(./sixteen --threads="$threads" --total="$games" --seed="$seed" --summary \
             --play="load=$ckpt ply=$ply $extra" 2>/dev/null | grep '^eval:')
    get() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<< "$line"; }
    printf "%-28s %3s %18s %22s %22s %22s %9s %9s %6s %7s\n" "$(basename "$ckpt")" "$ply" \
      "$(get mean) ± $(get ci95)" "$(get r8192)" "$(get r16384)" "$(get r32768)" \
      "$(get move_us)" "$(get max_move_ms)" "$(get 'over_[0-9]*ms')" "$(get avg_ply)"
  done
done
