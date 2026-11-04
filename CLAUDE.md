# sixteen

- `make` / `make test` - build; tests take ~25 s and ~340 MB (one test builds the full default network)
- `./eval.sh -n 1000 -p "1 2 3" a.ckpt b.ckpt` - compare checkpoints on the same seeded games; time moves with 1 thread and nothing else running
- Any new program must call `InitLookUpTables()` before using `Board64`, or slides silently do nothing
- Keep the checkpoint format and tuple index layout stable: the Kaggle chain resumes from the latest checkpoint
- Fast paths (`NTupleNetwork::GetValue`, `Search.h`) must stay bit-identical to the `*Reference` / plain expectimax code; tests compare them exactly
- Verify speedups end to end: old vs new binary, `--threads=1 --evil="seed=42"`, from the same checkpoint, then `cmp` the saved checkpoints
- After writing a test, plant the bug it targets and confirm it fails (this caught several tests that could never fail)
- Kaggle sessions build from `main` on GitHub (itsmaxhuynh/sixteen): commit and push to reach them; `git pull` first, since the chain bot commits `kaggle/chain-state`
- `kaggle/chain.sh` runs from `.github/workflows/chain.yml` (cron every 30 min, but GitHub drops most runs: `gh workflow run chain.yml` to chain now), once per chain (main, x3a, x3b, x4a, x4b, x5); to resume a halted chain, clear `halted=` in `kaggle/chain-state` (experiments: `chain-state-<exp>`)
- Live Kaggle log: `timeout 60 kaggle kernels logs -f itsmaxhuynh/sixteen-train > live.log` (`output` is empty until the run ends)
- `kaggle kernels output itsmaxhuynh/sixteen-train/<v> --file-pattern 'sixteen\.ckpt'` - fetch a finished version's checkpoint (~545 MB); the running version has no output yet (`kaggle kernels files .../<v>` lists them)
- Heavy training runs on Kaggle; keep local runs small
- Stages k > 0 train from boards collected at their split, so their tile table always shows 16384 at 100%; read the `splits` line instead
- 16k reach rate (stage 0 only): `./sixteen --threads=2 --total=1000 --collect=probe.bin --collect-count=1000000 --play="load=ckpt stage=0 ply=3"` (~0.9 games/s at 3-ply)
