#!/bin/bash
# Build the Kaggle kernel (sources embedded in train.py) and push it, which starts a run.
#
# usage: [EXP=name] [MIN_GAMES=n] kaggle/push.sh <kaggle-username> [hours]
#   EXP: experiment (kernel sixteen-<EXP> with its own config, see below); unset = the main run,
#        kernel sixteen-train
#   hours: budget per session, default 11 (Kaggle CPU sessions stop at 12h)
#   SEED: an extra kernel to resume from on the first push of an experiment (e.g. sixteen-train, to
#         start from the main run's checkpoint); later pushes list only the kernel itself
#   exit status: 0 pushed, 75 all of Kaggle's batch CPU sessions are busy (retry later), else failed
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
exp="${EXP:-}"
slug="sixteen-${exp:-train}"
out="kaggle/build${exp:+-$exp}"

mkdir -p "$out"
sources=$(tar czf - makefile *.h sixteen.cpp | base64 -w0)
# Per stage (list index = stage): alpha schedule ('step' as in [MS-TD], or 'cos' annealing down to
# alpha_min) and the value every board starts at when the stage is created (0 = [MS-TD]'s zero init,
# e.g. 320000 = optimistic init). Only stages that have not started yet should be changed.
# Cosine in every stage. The main run's stage 0 trained with 'step' and is finished; x4a/x4b inherit
# that stage 0 and start their fine-tune games after it, so changing it here affects new runs only.
anneal="['cos', 'cos', 'cos', 'cos', 'cos']"
init="[0, 0, 0, 0, 0]"
config="{'stages': 5, 'lambda': 0.5, 'games_per_stage': 10000000, 'decay_at': 5000000, 'collect': 100000, 'hours': $hours, 'margin_minutes': 20, 'threads': None, 'block': 10000, 'save_every': 200000, 'min_games': ${MIN_GAMES:-0}, 'anneal': $anneal, 'alpha_min': 0.000025, 'init': $init}"

# Experiments: the main config with some keys replaced.
# x3a / x3b: arXiv:2111.11090's best design, OTD+TC on the 8x6-tuple network with 2 stages (split
# at 16384) and V_init = 320k, plus our scalar features: TD for 90% of each stage's games, then TC.
# alpha is per state in the paper (0.1 for TD, 1.0 for TC), spread over the 69 weights a board
# reads (8 tuples x 8 symmetries + 5 scalar features). x3a keeps the TD alpha fixed (the paper);
# x3b anneals it with a cosine down to 0.01 per state.
case "$exp" in
  "") override="{}" ;;
  x3a|x3b)
    schedule=$([ "$exp" = x3a ] && echo tdtc || echo costc)
    override="{'stages': 2, 'patterns': '8x6', 'anneal': ['$schedule', '$schedule'], 'init': [320000, 320000], 'tc_fraction': 0.1, 'alpha_td': $(python3 -c 'print(0.1 / 69)'), 'alpha_td_min': $(python3 -c 'print(0.01 / 69)'), 'alpha_tc': $(python3 -c 'print(1.0 / 69)')}" ;;
  # x4a / x4b: hard starts (HardStarts.h) on top of the main run, from its checkpoint (first push with
  # SEED=sixteen-train MIN_GAMES=<its games>). Stage 0 gets 2M more games at its final alpha, then the
  # driver carries on with stage 1 and the rest as in main. Half the games start from boards rewound
  # up to 500 slides before the end of a game; x4b also doubles a tile of 30% of the start boards.
  x4a) override="{'stage_games': [12000000], 'hard': 0.5, 'hard_rewind': 500}" ;;
  x4b) override="{'stage_games': [12000000], 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3}" ;;
  # x5: the main method (5 stages, TD(0.5), cosine alpha, zero init) on the 8x6-tuple network,
  # trained from scratch with hard starts and tile-upgrading as in x4b, 10M games per stage like main
  # (x5's first session ran at the fixed 0.0025, close to the top of the cosine curve it then joins).
  # 8x6 runs at about a third of 4x6's moves/s, so this is the slowest chain.
  x5) override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3}" ;;
  *) echo "unknown EXP=$exp" >&2; exit 1 ;;
esac
config=$(python3 -c "import ast, sys; c = ast.literal_eval(sys.argv[1]); c.update(ast.literal_eval(sys.argv[2])); print(repr(c))" "$config" "$override")

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
if [ -n "${SEED:-}" ]; then
  self_source="${self_source:+$self_source, }\"$user/$SEED\""
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

# the CLI exits 0 even when Kaggle refuses the push, so read its message
msg=$(kaggle kernels push -p "$out" 2>&1) || true
echo "$msg"
if ! grep -q "successfully pushed" <<< "$msg"; then
  # all batch CPU sessions busy: try again later (exit 75 = EX_TEMPFAIL, chain.sh retries next run)
  grep -qi "session count" <<< "$msg" && exit 75
  exit 1
fi
echo "status: kaggle kernels status $user/$slug"
echo "output: kaggle kernels output $user/$slug -p <dir>"
