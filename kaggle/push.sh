#!/bin/bash
# Build the Kaggle kernel (sources embedded in train.py) and push it, which starts a run.
#
# usage: [MIN_GAMES=n] kaggle/push.sh <kaggle-username> [hours]
#   hours: budget per session, default 11 (Kaggle CPU sessions stop at 12h)
#   MIN_GAMES: the session refuses to train unless it finds a checkpoint with at least this many
#              games in total (kaggle/chain.sh sets it), so a broken chain never restarts from scratch
#
# Each run resumes the multi-stage schedule (see kaggle/train_template.py) from the newest
# sixteen.ckpt and boards*.bin under /kaggle/input. The kernel lists its own output as a source,
# so a new run picks up where the previous version finished.
#
# Schedule (CONFIG below), after [MS-TD]: 5 stages (Strategy 3), 100,000 boards collected per split,
# lambda 0.5. games_per_stage/decay_at follow Fig. 17 (10M games, alpha cut at 5M), which the paper
# shows for the first stage; using the same numbers for every stage is our choice.
set -euo pipefail
cd "$(dirname "$0")/.."

user="${1:?usage: kaggle/push.sh <kaggle-username> [hours]}"
hours="${2:-11}"
slug="sixteen-train"
out="kaggle/build"

mkdir -p "$out"
sources=$(tar czf - makefile *.h sixteen.cpp | base64 -w0)
config="{'stages': 5, 'lambda': 0.5, 'games_per_stage': 10000000, 'decay_at': 5000000, 'collect': 100000, 'hours': $hours, 'margin_minutes': 20, 'threads': None, 'block': 10000, 'save_every': 200000, 'min_games': ${MIN_GAMES:-0}}"

python3 - "$sources" "$config" > "$out/train.py" <<'EOF'
import sys
tpl = open("kaggle/train_template.py").read()
print(tpl.replace("@SOURCES@", sys.argv[1]).replace("@CONFIG@", sys.argv[2]), end="")
EOF

# resume from the previous version's output; the very first push has no output to read yet
if kaggle kernels status "$user/$slug" > /dev/null 2>&1; then
  self_source="\"$user/$slug\""
else
  echo "first push of $user/$slug: no previous output to resume from"
  self_source=""
fi

cat > "$out/kernel-metadata.json" <<EOF
{
  "id": "$user/$slug",
  "title": "$slug",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [$self_source]
}
EOF

kaggle kernels push -p "$out"
echo "status: kaggle kernels status $user/$slug"
echo "output: kaggle kernels output $user/$slug -p <dir>"
