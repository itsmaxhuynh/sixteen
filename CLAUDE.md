# sixteen

- `make` / `make test` - build; tests take ~25 s and ~340 MB (one test builds the full default network)
- `./eval.sh -n 1000 -p "1 2 3" a.ckpt b.ckpt` - compare checkpoints on the same seeded games; time moves with 1 thread and nothing else running
- Any new program must call `InitLookUpTables()` before using `Board64`, or slides silently do nothing
- Keep the checkpoint format and tuple index layout stable: the Kaggle chain resumes from the latest checkpoint
- Fast paths (`NTupleNetwork::GetValue`, `Search.h`) must stay bit-identical to the `*Reference` / plain expectimax code; tests compare them exactly
- Verify speedups end to end: old vs new binary, `--threads=1 --evil="seed=42"`, from the same checkpoint, then `cmp` the saved checkpoints
- After writing a test, plant the bug it targets and confirm it fails (this caught several tests that could never fail)
- Kaggle sessions build from `main` on GitHub (itsmaxhuynh/sixteen): commit and push to reach them; `git pull` first, since the chain bot commits `kaggle/chain-state`
- `kaggle/chain.sh` runs from `.github/workflows/chain.yml` every 30 min; to resume a halted chain, clear `halted=` in `kaggle/chain-state`
- Live Kaggle log: `timeout 60 kaggle kernels logs -f itsmaxhuynh/sixteen-train > live.log` (`output` is empty until the run ends)
- Heavy training runs on Kaggle; keep local runs small
