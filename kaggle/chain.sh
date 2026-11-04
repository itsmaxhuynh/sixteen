#!/bin/bash
# Chain Kaggle sessions: when the current session has finished, verify its checkpoint and push the
# next one. Meant to run every 30 minutes from GitHub Actions (.github/workflows/chain.yml), but works
# from any shell with the kaggle CLI logged in.
#
# usage: [EXP=name] kaggle/chain.sh <kaggle-username>
#   EXP: chain the experiment kernel sixteen-<EXP> (state in chain-state-<EXP>, see push.sh);
#        unset = the main run, sixteen-train
#
#   RUNNING / QUEUED   nothing to do
#   COMPLETE           download the output, check that the checkpoint loads and that training moved
#                      forward (more games, or more collected boards), then push the next session with
#                      MIN_GAMES set so it refuses to start without that checkpoint
#   all stages trained stop chaining (state: halted=done)
#   no free session    (push.sh exits 75) nothing recorded, the next run tries again
#   anything else      push nothing, record why (halted=<reason>) and exit 1 so someone looks
#   (a failed status call or download only fails this run; the next run tries again)
#
# Once halted, later runs do nothing until kaggle/chain-state is edited (clear `halted=`).
# State lives in kaggle/chain-state (key=value, committed by the workflow); a history line per push or
# halt goes to kaggle/chain-history.
#
# For tests: KAGGLE (kaggle CLI), PUSH (push script), NOW (epoch seconds) and CHAIN_DIR (where state,
# history and the downloaded output go) can be overridden.
set -euo pipefail
cd "$(dirname "$0")/.."

user="${1:?usage: kaggle/chain.sh <kaggle-username>}"
exp="${EXP:-}"
slug="$user/sixteen-${exp:-train}"
KAGGLE="${KAGGLE:-kaggle}"
PUSH="${PUSH:-kaggle/push.sh}"
NOW="${NOW:-$(date +%s)}"
dir="${CHAIN_DIR:-kaggle}"
state="$dir/chain-state${exp:+-$exp}"
history="$dir/chain-history${exp:+-$exp}"
out="$dir/runs/${exp:-latest}"
stall_hours=13 # a pushed session runs at most 12h; equal progress after this long means it made none

# ---- state: last_games, last_boards, pushed_at, pushes, halted
last_games=0 last_boards=0 pushed_at=0 pushes=0 halted=""
if [ -f "$state" ]; then
  while IFS='=' read -r k v; do
    case "$k" in
      last_games|last_boards|pushed_at|pushes) [[ "$v" =~ ^[0-9]+$ ]] && printf -v "$k" '%s' "$v" ;;
      halted) halted="$v" ;;
    esac
  done < "$state"
fi

save_state() {
  mkdir -p "$dir"
  printf 'last_games=%s\nlast_boards=%s\npushed_at=%s\npushes=%s\nhalted=%s\n' \
    "$last_games" "$last_boards" "$pushed_at" "$pushes" "$halted" > "$state"
}

note() { echo "$(date -u -d "@$NOW" '+%Y-%m-%d %H:%M UTC') $*" | tee -a "$history"; }

halt() {
  halted="$1"
  save_state
  note "HALT: $1"
  echo "halted=$(tr '\n\r' '  ' <<< "$1")" >> "${GITHUB_OUTPUT:-/dev/null}"
  exit 1
}

if [ -n "$halted" ]; then
  echo "chain is halted ($halted); clear halted= in $state to resume"
  exit 0
fi

# ---- where is the current session?
# a failed status call or download may be a network blip: fail this run (retried next time), no halt
status_line=$($KAGGLE kernels status "$slug" 2>&1) || { echo "kaggle status failed: $status_line"; exit 1; }
status=$(sed -n 's/.*KernelWorkerStatus\.\([A-Z_]*\).*/\1/p' <<< "$status_line")
echo "kaggle: $slug is ${status:-unknown} ($status_line)"
case "$status" in
  RUNNING|QUEUED) exit 0 ;;
  COMPLETE) ;;
  *) halt "session status ${status:-unknown}: $status_line" ;;
esac

# ---- finished: fetch and check its output
rm -rf "$out"
mkdir -p "$out"
$KAGGLE kernels output "$slug" -p "$out" > /dev/null 2>&1 || { echo "downloading the output failed"; exit 1; }
[ -f "$out/sixteen.ckpt" ] || halt "the finished session left no sixteen.ckpt"

make -s sixteen
info=$(./sixteen --info="$out/sixteen.ckpt" 2>&1 | tail -1) || halt "checkpoint does not load: $info"
games_list=$(sed -n 's/.* games=\([0-9,]*\).*/\1/p' <<< "$info")
[ -n "$games_list" ] || halt "cannot read games from: $info"
# --info reads only the header; loading every weight (and playing one game) catches a truncated file
./sixteen --total=1 --play="load=$out/sixteen.ckpt" > /dev/null 2>&1 || halt "checkpoint does not fully load (truncated?)"
games=$(awk -F, '{ s = 0; for (i = 1; i <= NF; ++i) s += $i; print s }' <<< "$games_list")
boards=0
for f in "$out"/boards*.bin; do
  [ -f "$f" ] && boards=$((boards + ($(stat -c %s "$f") - 12) / 12))
done
echo "output: $info, boards=$boards (last push: games=$last_games boards=$last_boards)"

if grep -qs "stages trained" "$out/train.log"; then
  halted=done
  save_state
  note "DONE: all stages trained, $info"
  echo "done=1" >> "${GITHUB_OUTPUT:-/dev/null}"
  exit 0
fi

[ "$games" -ge "$last_games" ] || halt "games went backwards: $games < $last_games"
if [ "$games" -eq "$last_games" ] && [ "$boards" -le "$last_boards" ]; then
  # either this is the output we already chained from (the pushed session has not finished yet),
  # or the pushed session finished without any progress
  age=$((NOW - pushed_at))
  if [ "$age" -lt $((stall_hours * 3600)) ]; then
    echo "same output as at the last push ${age}s ago; the pushed session has not finished yet"
    exit 0
  fi
  halt "no progress: games=$games boards=$boards, same as the push ${age}s ago"
fi

# ---- push the next session; it must find at least these games
rc=0
EXP="$exp" MIN_GAMES="$games" $PUSH "$user" 11 || rc=$?
if [ "$rc" -eq 75 ]; then
  echo "no free Kaggle CPU session; the next run tries again"
  exit 1
fi
[ "$rc" -eq 0 ] || halt "push failed (exit $rc)"
last_games=$games last_boards=$boards pushed_at=$NOW pushes=$((pushes + 1))
save_state
note "PUSH #$pushes: from games=$games ($games_list) boards=$boards"
echo "pushed=1" >> "${GITHUB_OUTPUT:-/dev/null}"
