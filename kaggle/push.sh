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
#   EXTRA: a StartPositions file packed into the sources as extra-starts.bin, for 'extra_share'
#          (break starts, see kaggle/train_template.py)
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
extra_dir=$(mktemp -d)
trap 'rm -rf "$extra_dir"' EXIT
if [ -n "${EXTRA:-}" ]; then
  cp "$EXTRA" "$extra_dir/extra-starts.bin"
  sources=$(tar czf - makefile *.h sixteen.cpp -C "$extra_dir" extra-starts.bin | base64 -w0)
else
  sources=$(tar czf - makefile *.h sixteen.cpp | base64 -w0)
fi
# Per stage (list index = stage): alpha schedule ('step' as in [MS-TD], or 'cos' annealing down to
# alpha_min) and the value every board starts at when the stage is created (0 = [MS-TD]'s zero init,
# e.g. 320000 = optimistic init). Only stages that have not started yet should be changed.
# Cosine in every stage. The main run's stage 0 trained with 'step' and is finished; x4a/x4b inherit
# that stage 0 and start their fine-tune games after it, so changing it here affects new runs only.
anneal="['cos', 'cos', 'cos', 'cos', 'cos']"
init="[0, 0, 0, 0, 0]"
config="{'stages': 5, 'lambda': 0.5, 'games_per_stage': 10000000, 'decay_at': 5000000, 'collect': 100000, 'hours': $hours, 'margin_minutes': 20, 'threads': None, 'block': 10000, 'save_every': 200000, 'min_games': ${MIN_GAMES:-0}, 'anneal': $anneal, 'alpha_min': 0.000025, 'init': $init, 'recollect': True}"

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
  # x5s1..x5s4: 10M more games for one stage of x5's finished checkpoint, each in its own kernel at
  # the same time (single-stage sessions, see kaggle/train_template.py), merged afterwards with
  # ./sixteen --merge. First push: SEED=sixteen-x5 MIN_GAMES=50000000. The stage's boards are first
  # topped up to 100k distinct boards, then it trains on a fresh cosine from 0.00025 (the paper's
  # rate after its decay) down to alpha_min, with x5's hard starts and tile-upgrading.
  x5s[1-4])
    k=${exp#x5s}
    targets=$(python3 -c "import sys; k = int(sys.argv[1]); print([20000000 if s == k else 10000000 for s in range(5)])" "$k")
    override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3, 'only_stage': $k, 'stage_games': $targets, 'restart_alpha': 0.00025, 'restart_from': 10000000}" ;;
  # x5t1..x5t4: 10M more again (20M -> 30M) for stage k, from x5s<k>'s finished checkpoint and boards,
  # with a second warm restart of the same cosine (0.00025 down to 2.5e-05). First push:
  # SEED=sixteen-x5s<k> MIN_GAMES=60000000. Merge as for x5s (base sixteen-x5, k=sixteen-x5t<k>).
  x5t[1-4])
    k=${exp#x5t}
    targets=$(python3 -c "import sys; k = int(sys.argv[1]); print([30000000 if s == k else 10000000 for s in range(5)])" "$k")
    override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3, 'only_stage': $k, 'stage_games': $targets, 'restart_alpha': 0.00025, 'restart_from': 20000000}" ;;
  # x5u1..x5u4: 10M more (30M -> 40M) for stage k, from x5t<k>'s finished checkpoint and boards, on a
  # cooler cosine: from x5t's final alpha 2.5e-05 down to 2.5e-06 (fine-tuning, no warm restart).
  # First push: SEED=sixteen-x5t<k> MIN_GAMES=70000000. Merge as for x5s (base sixteen-x5, k=sixteen-x5u<k>).
  x5u[1-4])
    k=${exp#x5u}
    targets=$(python3 -c "import sys; k = int(sys.argv[1]); print([40000000 if s == k else 10000000 for s in range(5)])" "$k")
    override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3, 'only_stage': $k, 'stage_games': $targets, 'restart_alpha': 0.000025, 'restart_from': 30000000, 'alpha_min': 0.0000025}" ;;
  # x5u0: stage 0 never got the extensions above (it stopped at x5's 10M), so it trains on its own
  # from x5's checkpoint: 10M more (10M -> 20M; ~30h at stage 0's ~340k games/h, so it fits before
  # TAAI) on one cosine from 0.00025 down to 2.5e-06, starting from the empty board as in x5.
  # First push: SEED=sixteen-x5 MIN_GAMES=50000000. Merge with base sixteen-x5, 0=sixteen-x5u0.
  x5u0) override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3, 'only_stage': 0, 'stage_games': [20000000, 10000000, 10000000, 10000000, 10000000], 'restart_alpha': 0.00025, 'restart_from': 10000000, 'alpha_min': 0.0000025}" ;;
  # x5v0..x5v4: the x5u runs again at a fixed alpha of 2.5e-05, no cosine. x5u's cosine went down to
  # 2.5e-06, and below ~1e-05 the float32 weights drop most updates (the scalar feature weights are
  # 30k-174k, where float32 steps are 0.004-0.016), so x5u2-x5u4 got worse late in training.
  # x5v1..x5v4 train stage k 30M -> 40M from x5t<k>'s finished checkpoint and boards. First push:
  # SEED=sixteen-x5t<k> MIN_GAMES=70000000. x5v0 trains stage 0 on to 20M from x5u0's first session
  # (12.65M games, still at alpha ~2.4e-04, undamaged). First push: SEED=sixteen-x5u0
  # MIN_GAMES=52652623. Merge with base sixteen-x5, k=sixteen-x5v<k> (0=sixteen-x5v0).
  x5v[0-4])
    k=${exp#x5v}
    targets=$(python3 -c "import sys; k = int(sys.argv[1]); print([(20000000 if k == 0 else 40000000) if s == k else 10000000 for s in range(5)])" "$k")
    # Start boards: recollected every session (recollect) with the newest undamaged earlier stages:
    # stage 0 from x5v0 (x5u0 until x5v0 has more games), stage j > 0 from x5v<j> (else x5t<j>);
    # x5u1-x5u4 are left out (float32-damaged late in training, see above).
    collect_from=$(python3 -c "import sys; k = int(sys.argv[1]); print({j: ['sixteen-x5v0', 'sixteen-x5u0'] if j == 0 else [f'sixteen-x5v{j}', f'sixteen-x5t{j}'] for j in range(k)})" "$k")
    override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3, 'only_stage': $k, 'stage_games': $targets, 'fixed_alpha': 0.000025, 'collect_from': $collect_from}" ;;
  # x5w1, x5w2: stage k again from x5t<k> (30M games) at x5v's fixed alpha, with a fifth of the games
  # started from the downgraded boards before the agent's corner breaks (EXTRA=<boards for stage k>, made
  # from recorded games at ply=3 use=0,1,2,2,2 downgrade=32768). No recollection: x5t<k>'s boards.
  # Push: EXP=x5w<k> SEED=sixteen-x5t<k> MIN_GAMES=70000000 EXTRA=<file>. Merge with base sixteen-x5.
  x5w[1-2])
    k=${exp#x5w}
    targets=$(python3 -c "import sys; k = int(sys.argv[1]); print([40000000 if s == k else 10000000 for s in range(5)])" "$k")
    [ -n "${EXTRA:-}" ] || { echo "EXP=$exp needs EXTRA=<boards file for stage $k>" >&2; exit 1; }
    override="{'patterns': '8x6', 'hard': 0.5, 'hard_rewind': 500, 'upgrade': 0.3, 'only_stage': $k, 'stage_games': $targets, 'fixed_alpha': 0.000025, 'recollect': False, 'extra_share': 0.2}" ;;
  *) echo "unknown EXP=$exp" >&2; exit 1 ;;
esac
config=$(python3 -c "import ast, sys; c = ast.literal_eval(sys.argv[1]); c.update(ast.literal_eval(sys.argv[2])); print(repr(c))" "$config" "$override")

# the sources go through a file: with EXTRA they can pass Linux's 128 KB limit on one argument
printf '%s' "$sources" > "$extra_dir/sources.b64"
python3 - "$extra_dir/sources.b64" "$config" > "$out/train.py" <<'EOF'
import sys
tpl = open("kaggle/train_template.py").read()
print(tpl.replace("@SOURCES@", open(sys.argv[1]).read()).replace("@CONFIG@", sys.argv[2]), end="")
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
# kernels whose checkpoints only collect start boards (collect_from, see train_template.py)
for src in $(python3 -c "import ast, sys; c = ast.literal_eval(sys.argv[1]); print(' '.join(sorted({s for v in c.get('collect_from', {}).values() for s in v} - {sys.argv[2], sys.argv[3]})))" "$config" "$slug" "${SEED:-}"); do
  self_source="${self_source:+$self_source, }\"$user/$src\""
done

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
