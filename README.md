# SAGE: Spectral Artifact-Guided Decision-Based Evasion of Diffusion-Generated Image Detectors

Reference implementation of protocol **SAGE-v1.0**: Spectral Artifact Profiling (SAP), Adaptive
Spectral Alignment (ASA) and Spectral-Biased Query Refinement (SBQR), the label-only baselines
(Boundary Attack, HSJA, QEBA-S) under one shared query counter, and the evaluation protocol
(primary comparison, component controls, SD-v1 → SD-v2 transfer).

SAGE is **training-free**: the GPU jobs are offline spectral profiling, clean screening and
attack evaluation against frozen detectors (UniversalFakeDetect, DIRE).

## Layout

```
configs/sage_v1.0.json      frozen protocol parameters (authoritative)
sage/
  spectral.py               unitary FFT, radial bands, T(x; alpha), Hermitian pairing, Eq. 12 sampling
  sap.py                    mean real / generated power spectra, prior I/O            (Eq. 3)
  asa.py                    band-normalised KL + lambda * LPIPS, grid coordinate descent (Eq. 7-8)
  weights.py                learned / uniform / low-frequency sampling weights        (Eq. 10-11)
  query.py                  QueryOracle (budget, cache, archive, checkpoints) + batched scheduler
  init_point.py             reference verification + boundary bisection                (Eq. 9)
  sbqr.py                   spectral-biased refinement with step adaptation            (Eq. 12-14)
  attack.py                 complete SAGE pipeline; controlled refinement variants
  baselines.py              BA (foolbox-3 defaults), HSJA (L2), QEBA-S (resize 4)
  detectors.py              UFD and DIRE adapters (DIRE recomputed on every query)
  perceptual.py             LPIPS(alex, v0.1), PSNR, RMSE
  data.py, runinfo.py       manifest access, execution-lock metadata
experiments/
  build_prior.py            SAP priors (mixed, SD-v1-only)
  screen_clean.py           clean eligibility / FPR (not charged to budgets)
  run_attacks.py            pilot | primary | controls | transfer, sharded + resumable
  analyze.py                Table 2, Table 3 A/B, ASR-vs-query figure, bootstrap CIs
scripts/umhpc/              environment setup and A100 self-check for UMHPC (DICC)
scripts/data/               manifest builder / verifier
slurm/                      single-A100 job scripts + submit_all.sh
tests/                      spectral invariants and toy-detector end-to-end tests
docs/                       CODE_PLAN.md (design), DATA.md (dataset preparation)
```

## Quick start on UMHPC

```bash
git clone git@github.com:williamm0123/SAGE.git ~/SAGE && cd ~/SAGE
export PROJECT_ROOT=$HOME/SAGE
bash scripts/umhpc/setup_env.sh          # login node: conda env, third-party code, public weights
# copy DIRE lsun_adm.pth to weights/dire/ and the DiffusionForensics subsets to data/raw/ (docs/DATA.md)
source scripts/umhpc/env.sh
python -m pytest -q tests
mkdir -p logs && sbatch scripts/umhpc/check_env.sbatch
python scripts/data/build_manifest.py ...   # see docs/DATA.md
bash slurm/submit_all.sh                 # prepare -> pilot; inspect, then:
bash slurm/submit_all.sh --full          # primary + controls/transfer -> analysis
```

Every GPU job requests a single A100 (`--partition=gpu-a100 --gres=gpu:1`). Results land in
`results/<experiment>/<detector>/<method>/shardXXXofYYY.jsonl` with a `.meta.json` execution lock
(config hash, manifest hash, prior hash, detector weight hashes, commits, GPU). Re-submitting a job
resumes where it stopped.

## Protocol notes

* Success = detector predicts Real **and** LPIPS(x, x_o) ≤ 0.05; every detector call during the
  attack (ASA verification, reference verification, bisection, refinement) counts toward the
  5,000-query budget; cached repeats are free; clean screening is reported separately.
* The scheduler batches pending queries from many concurrent attacks into one detector call. This
  does not change any per-attack query count; it only improves GPU throughput (important for DIRE,
  where each query runs 20-step DDIM inversion + 20-step DDIM sampling).
* Baselines start from the same reference pool, order and bisection as SAGE.
