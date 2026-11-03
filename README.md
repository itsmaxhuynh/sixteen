# sixteen

A 2048 agent for TAAI, named after the 2^16 tile. Ported from the Threes agent in
`../Threes-Puzzle-AI` and checked against the paper it implements:
Yeh et al., *Multi-Stage Temporal Difference Learning for 2048-like Games*
(https://arxiv.org/abs/1606.07374), cited as [MS-TD] in the code.

## Build and test

```sh
make          # ./sixteen
make test     # board vs. naive 2048, symmetries, split rules, scalar features, eq. (6), checkpoint,
              # fast evaluation vs. reference, cached / timed search vs. plain expectimax
```

## What follows the paper

| [MS-TD] | Where |
|---|---|
| 4 six-tuples of Fig. 4(b), 8 symmetries each | `NTupleNetwork::DefaultPatterns` |
| Large-tile feature (n2048..n32k), Sec. IV.A | `kFeatLargeTile` |
| Empty / distinct / mergeable / v-2v neighbour features, Sec. IV.E (a) | `kFeatEmpty` ... `kFeatNeighbor` |
| Afterstate TD, Δθ = αδ on every occurring feature, eq. (5) | `TdPlayer::Learn` |
| α = 0.0025, then 0.00025 once saturated, Sec. IV.E (b) | Kaggle schedule |
| TD(λ), λ=0.5, returns truncated at 5 steps, eq. (6) | `TdPlayer::LambdaReturn` |
| Updates after the game ends, Sec. IV.E | `TdPlayer::Learn` |
| Stages split at T16k, T16+8k, T16+8+4k, T16+8+4+2k (Strategy 3) | `SplitLevel`, `stages=5` |
| Stage k trained from 100,000 boards collected at its split, zero init, earlier stages frozen | `--collect`, `--start`, `stage=` |
| n-ply expectimax: 1-ply greedy, chance nodes 0.9/0.1 over empty cells, Fig. 2 | `ply=` |

Not specified by the paper, so these are our choices: the update order (backward over the game),
using the Fig. 17 schedule (10M games, α cut at 5M) for every stage, and the uniform pick of
collected boards during collection (training goes round robin).

## Commands

```sh
# stage 0 from scratch
./sixteen --learn --threads=4 --total=5000000 --save-every=200000 --play="save=w.ckpt"
# boards at the first split (T16k)
./sixteen --collect=boards1.bin --collect-count=100000 --total=1000000000 --play="load=w.ckpt"
# stage 1 from those boards (starts at zero)
./sixteen --learn --total=5000000 --start=boards1.bin --play="load=w.ckpt save=w.ckpt stage=1"
# ... repeat collect/train for stages 2..4, collecting with --start=boards<k>.bin

./sixteen --info=w.ckpt                                              # stage, games per stage
./sixteen --total=1000 --seed=1 --summary --play="load=w.ckpt ply=3"  # evaluate
./eval.sh -n 1000 -p "1 2 3" a.ckpt b.ckpt                            # compare on the same games
```

`--seed=N` gives game i the spawn seed N+i, so every run plays the same games whatever the thread
count. With `--summary` an `eval:` line adds the mean with a 95% interval, 8192/16384/32768 reach
rates with 95% Wilson intervals, mean and max time per move, moves over `--move-limit-ms` (default
100) and the average search depth. Measure move times with 1 thread.

## Match play (TAAI: 100 ms per move)

`--play="load=w.ckpt time=50"` searches 1, 2, 3, ... plies within 50 ms per move (`Search.h`) and
plays the move of the deepest search that finished. A depth starts only if its cost, predicted
from the growth between the last two depths, fits the budget; an unfinished depth is dropped at the
hard deadline (`hard=`, default = `time`). Crowded boards get searched deeper. Chance-node values
go into a position cache (`tt=22`: 2^22 entries, 64 MB) keyed by board, plies left and stage, so
the cached search returns exactly what plain expectimax would. `prune=p` (off by default) stops
searching spawn outcomes whose path probability is below p.

All of these take `--threads=N` (default 1). Training is Hogwild: one shared network, lock-free
updates through relaxed `std::atomic_ref`, a pause at each `--save-every` boundary so checkpoints are
consistent. On 4 threads the 20k-game learning curve matches 1 thread at 3.3x the speed, and
ThreadSanitizer is clean.

`--play` keys: `load= save= alpha= lambda= ply= stage= time= hard= maxply= tt= prune=`; `stages=` and `features=` (all, none or a
bitmask) apply only when starting fresh. Runs stop on `--total`, `--time-limit`, SIGTERM or Ctrl-C
and always save.

## Train on Kaggle

```sh
pip install kaggle   # token in ~/.kaggle/kaggle.json
kaggle/push.sh <kaggle-username> [hours=11]
```

`push.sh` embeds the sources into one kernel script and pushes it, which starts a session. The kernel
runs the whole schedule (train stage k, collect boards, start stage k+1) and resumes from the
checkpoint and board files of its previous version's output. Push again to continue.

## Design notes

* Board: `uint64_t`, 4 bits per cell holding the rank (tile = 2^rank), row r at bits 16r.
  Directions `0=U 1=R 2=D 3=L`. Slides are XOR-delta row tables, rewards are per-row merge tables.
* Memory: 260 MB per stage, 1.3 GB with all 5 stages. Stages are allocated when first used.
* Checkpoint v2: header (patterns, features, stages, active stage, games per stage, α, λ) and raw
  floats for initialized stages; written to `.tmp` and renamed.

## Known limits

* 4-bit cells stop at 32768: two 32768 tiles do not merge, so 65536 cannot be built yet. If the
  real game merges them, our board copy in a match would drift from the real one.
* The arena shell keeps the CGI protocol from the Threes project; the TAAI protocol is unchecked.
