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
# 16384 / 32768 with 95% intervals, and the average / max time per move. Then, per run, the full
# breakdown: every tile's reach rate with its interval and the share of games that ended on it,
# and the reach rates of the stage splits (16k, 16+8k, 16+8+4k, 16+8+4+2k).
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
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 1; }

make -s sixteen

# breakdown <title> <eval line>: the detail block of one run
breakdown() {
  awk -v title="$1" '{
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^tile[0-9]+=/ || $i ~ /^split/) {
        split($i, kv, "="); key = kv[1]; v = kv[2]
        p = v; sub(/%.*/, "", p); ci = v; sub(/^[^[]*/, "", ci)
        if (key ~ /^tile/) { t = substr(key, 5); tiles[++nt] = t; reach[t] = p; tci[t] = ci }
        else { s = substr(key, 6); splits[++ns] = s; sreach[s] = p; sci[s] = ci }
      }
    }
  } END {
    print "reach  " title
    printf "  %-17s %8s  %-17s %9s\n", "tile", "reach", "ci95", "ended on"
    first = nt
    for (k = 1; k <= nt; k++) {
      next_reach = k < nt ? reach[tiles[k + 1]] : 0
      if (reach[tiles[k]] - next_reach > 0) { first = k; break }
    }
    for (k = first; k <= nt; k++) {
      t = tiles[k]; next_reach = k < nt ? reach[tiles[k + 1]] : 0
      if (reach[t] == 0 && k > first) break
      printf "  %-17s %7.2f%%  %-17s %8.2f%%\n", t, reach[t], tci[t], reach[t] - next_reach
    }
    for (k = 1; k <= ns; k++) printf "  %-17s %7.2f%%  %s\n", "split " splits[k], sreach[splits[k]], sci[splits[k]]
  }' <<< "$2"
}
# progress <games> <label>: passes sixteen's output through, and for each block report (one every
# `step` games, before the final summary) prints "N/games done, elapsed, ~left" to stderr
progress() {
  local total=$1 label=$2 start=$SECONDS summary=0 l n el
  while IFS= read -r l; do
    [[ $l == "summary of all games:"* ]] && summary=1
    if (( !summary )) && [[ $l =~ ^([0-9]+)$'\t'avg ]]; then
      n=${BASH_REMATCH[1]} el=$((SECONDS - start))
      printf 'progress %s: %d/%d games (%d%%), %dm elapsed, ~%dm left\n' "$label" "$n" "$total" \
        $((100 * n / total)) $((el / 60)) $((el * (total - n) / n / 60)) >&2
    fi
    printf '%s\n' "$l"
  done
}
step=$(( games / 20 > 0 ? games / 20 : 1 ))  # a progress line every 5% of the games
details=""
printf "%-28s %3s %18s %22s %22s %22s %9s %9s %6s %7s\n" checkpoint ply "mean ± ci95" "8192 %" "16384 %" "32768 %" "move_us" "max_ms" "over" "avg_ply"
for ckpt in "$@"; do
  for ply in $plies; do
    line=$(./sixteen --threads="$threads" --total="$games" --seed="$seed" --summary --block="$step" \
             --play="load=$ckpt ply=$ply $extra" 2>/dev/null |
           progress "$games" "$(basename "$ckpt") ply=$ply${extra:+ $extra}" | grep '^eval:')
    get() { sed -n "s/.* $1=\([^ ]*\).*/\1/p" <<< "$line"; }
    printf "%-28s %3s %18s %22s %22s %22s %9s %9s %6s %7s\n" "$(basename "$ckpt")" "$ply" \
      "$(get mean) ± $(get ci95)" "$(get r8192)" "$(get r16384)" "$(get r32768)" \
      "$(get move_us)" "$(get max_move_ms)" "$(get 'over_[0-9]*ms')" "$(get avg_ply)"
    details+=$(breakdown "$(basename "$ckpt") ply=$ply${extra:+ $extra} ($games games)" "$line")$'\n\n'
  done
done
# after the table: per run, every tile's reach rate (largest tile at least that) with its 95% interval
# and the share of games that ended on it, from the smallest tile any game ended on; then the stage splits
printf "\n%s" "$details"
