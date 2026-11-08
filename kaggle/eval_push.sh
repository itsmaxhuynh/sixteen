#!/bin/bash
# Push the Kaggle eval kernel (sixteen-eval): eval.sh on the newest checkpoint of each listed kernel.
#
# usage: [PLIES="1 2 3"] [GAMES=1000] [SEED=1] [THREADS=1] [EXTRA="play args"] \
#        kaggle/eval_push.sh <kaggle-username> <kernel> [kernel ...]
#   kernel: training kernels whose output (sixteen.ckpt) to evaluate, e.g. sixteen-train sixteen-x4b
#   THREADS: 1 keeps the move times meaningful (see eval.sh)
#   NAME: kernel slug to push (default sixteen-eval); another name runs beside a busy one
#   RUNS: instead of PLIES/EXTRA/GAMES, a list of "ply:extra:games" separated by ';', e.g.
#         RUNS="3:prune=0.001:1000;4:prune=0.001:300" (extra may be empty: "3::1000")
#   MERGE: evaluate one checkpoint merged from stages trained separately instead (./sixteen --merge),
#          "<base kernel> <stage>=<kernel> ...", e.g. MERGE="sixteen-x5 1=sixteen-x5s1 4=sixteen-x5s4";
#          its kernels are the sources, so no kernel arguments are needed
# Results: kaggle kernels output <user>/$NAME -p <dir> (eval.txt), or the live log.
set -euo pipefail
cd "$(dirname "$0")/.."

user="${1:?usage: kaggle/eval_push.sh <kaggle-username> <kernel> [kernel ...]}"
shift
merge="None"
if [ -n "${MERGE:-}" ]; then
  read -r -a words <<< "$MERGE"
  set -- "${words[0]}"
  parts=""
  for w in "${words[@]:1}"; do
    [[ "$w" =~ ^([0-9]+)=(.+)$ ]] || { echo "MERGE: expected <stage>=<kernel>, got $w" >&2; exit 1; }
    parts+="(${BASH_REMATCH[1]}, '${BASH_REMATCH[2]}'), "
    set -- "$@" "${BASH_REMATCH[2]}"
  done
  [ -n "$parts" ] || { echo "MERGE: no <stage>=<kernel> parts" >&2; exit 1; }
  merge="{'base': '${words[0]}', 'parts': [${parts%, }]}"
fi
[ $# -ge 1 ] || { sed -n '2,13p' "$0"; exit 1; }
name="${NAME:-sixteen-eval}"
out="kaggle/build-${name#sixteen-}"
mkdir -p "$out"

sources=$(tar czf - makefile *.h sixteen.cpp eval.sh | base64 -w0)
plies=$(python3 -c "import sys; print([int(p) for p in sys.argv[1].split()])" "${PLIES:-1 2 3}")
runs=$(python3 -c "import sys; print([(int(p), x, int(g)) for p, x, g in (r.split(':') for r in sys.argv[1].split(';') if r)])" "${RUNS:-}")
config="{'runs': $runs, 'plies': $plies, 'games': ${GAMES:-1000}, 'seed': ${SEED:-1}, 'threads': ${THREADS:-1}, 'extra': '${EXTRA:-}', 'merge': $merge}"
python3 - "$sources" "$config" > "$out/eval.py" <<'EOF'
import sys
tpl = open("kaggle/eval_template.py").read()
print(tpl.replace("@SOURCES@", sys.argv[1]).replace("@CONFIG@", sys.argv[2]), end="")
EOF

kernels=$(for k in "$@"; do printf '"%s/%s", ' "$user" "$k"; done | sed 's/, $//')
cat > "$out/kernel-metadata.json" <<EOF
{
  "id": "$user/$name",
  "title": "$name",
  "code_file": "eval.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [$kernels]
}
EOF

[ -n "${DRY:-}" ] && exit 0  # DRY=1: build only, do not push
msg=$(kaggle kernels push -p "$out" 2>&1) || true
echo "$msg"
grep -q "successfully pushed" <<< "$msg" || exit 1
echo "status: kaggle kernels status $user/$name"
