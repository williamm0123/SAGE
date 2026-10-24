# SAGE: Spectral Artifact-Guided Decision-Based Evasion of Diffusion-Generated Image Detectors

Reference implementation of protocol **SAGE-v1.0**: Spectral Artifact Profiling (SAP), Adaptive
Spectral Alignment (ASA) and Spectral-Biased Query Refinement (SBQR), the label-only baselines
(Boundary Attack, HSJA, QEBA-S) under one shared query counter, and the evaluation protocol
(primary comparison, component controls, SD-v1 → SD-v2 transfer).

SAGE is **training-free**: the GPU jobs are offline spectral profiling, clean screening and
attack evaluation against frozen detectors (UniversalFakeDetect, AEROBLADE).

## Layout

```
configs/sage_v1.0.json      frozen protocol parameters (authoritative)
sage/
  spectral.py               unitary FFT, radial bands, T(x; alpha), Hermitian pairing, Eq. 12 sampling
  sap.py                    mean real / generated power spectra, prior I/O            (Eq. 3)
  asa.py                    band-normalised KL + lambda * LPIPS, grid coordinate descent (Eq. 7-8)
  weights.py                learned / uniform / low-frequency sampling weights        (Eq. 10-11)
  query.py                  QueryOracle (budget, cache, archive, checkpoints), QueryLog, batched scheduler
  experiment.py             expected runs, crash-safe JSONL, execution locks, resumable runs
  init_point.py             reference verification + boundary bisection                (Eq. 9)
  sbqr.py                   spectral-biased refinement with step adaptation            (Eq. 12-14)
  attack.py                 complete SAGE pipeline; controlled refinement variants
  baselines.py              BA (foolbox-3 defaults), HSJA (L2), QEBA-S (resize 4)
  detectors.py              UFD and AEROBLADE adapters (DIRE adapter kept, not used - see below)
  perceptual.py             LPIPS(alex, v0.1), PSNR, RMSE
  data.py, runinfo.py       manifest access, execution-lock metadata
experiments/
  build_prior.py            SAP priors (mixed, SD-v1-only)
  screen_clean.py           clean eligibility / FPR (not charged to budgets)
  run_attacks.py            pilot | primary | controls | transfer, sharded + resumable
  status.py                 completeness vs eligible x seeds, throughput, projected GPU hours
  analyze.py                Table 2/3, paired + common-success comparisons, diagnostics, figures
scripts/umhpc/              environment setup and A100 self-check for UMHPC (DICC)
scripts/data/               manifest builder / verifier
slurm/                      single-A100 job scripts + submit_all.sh
tests/                      spectral invariants and toy-detector end-to-end tests
docs/                       CODE_PLAN.md (design), DATA.md (dataset preparation)
```

## Quick start on UMHPC

```bash
cd /scr/user/eric123/project/SAGE && git pull     # PROJECT_ROOT is derived from the repo location
bash scripts/umhpc/setup_env.sh          # login node: conda env, third-party code, public weights
source scripts/umhpc/env.sh              # DIRE_ROOT defaults to /scr/user/eric123/datasets/DIRE
python -m pytest -q tests
mkdir -p logs && sbatch scripts/umhpc/check_env.sbatch   # GPU, weights, UFD/AEROBLADE query determinism
bash slurm/submit_all.sh                 # prepare (manifest, priors, AEROBLADE threshold, screening) -> pilot
python experiments/status.py --data data/sage_v1 --results results   # pilot throughput + projected GPU-h
bash slurm/submit_all.sh --full          # primary + controls/transfer; analysis starts when all runs exist
```

## Jobs, time limit and resumption

* Every GPU job requests one A100 (`--partition=gpu-a100 --gres=gpu:1`) for at most `3-00:00:00`.
  With `GPU_SLOTS=1` (default) all GPU jobs are named `sage-gpu` and use `--dependency=singleton`,
  so exactly one runs at a time. Submit from the repo root.
* An attack job stops 30 min before its limit (`--stop-margin-min`) and exits 75; the sbatch wrapper
  then resubmits the same array task (`SAGE_AUTO_RESUBMIT`, at most `SAGE_MAX_RESUBMIT=30` times).
* Unfinished attacks keep a query log (`qlog/<id>_seed<s>.qlog`, 17 bytes per query). A resumed run is
  re-executed from its seed and answered from the log without new detector calls (still charged),
  so it continues exactly where it stopped; if the replayed image ever differs, the run restarts from
  scratch and is marked `replay_restart`.
* Finished runs are read from all shard files of a directory, so `NSHARD` / `NSHARD_CT` may change
  between submissions. Controls variants of one run are written in one fsync'ed write; a torn last
  line is repaired.

## Result integrity

* `results/<experiment>/<detector>/<method>/LOCK.json` freezes config, manifest, screening, prior
  content hash, detector weights / settings / batch, code hash, third-party commits, torch/CUDA and
  the baseline parameter dictionary. A later session with any difference aborts (exit 3). Clean
  screening must come from the same manifest, weights, settings and batch shape.
* Debug runs (`--limit`) and fixed-budget runs (`--fixed-budget`) use separate directories.
* `experiments/status.py` compares stored runs with eligible targets x locked seeds; `analyze.py`
  refuses incomplete or inconsistent results (`--allow-incomplete` gives a flagged draft; missing runs
  are never counted as failures). Throughput comes from `sessions.jsonl` (process wall time, images
  sent), not from summing overlapping per-run wall times.

## Protocol notes

* Success = detector predicts Real **and** LPIPS(x, x_o) ≤ 0.05; every detector call during the
  attack (ASA verification, reference verification, bisection, refinement) counts toward the
  5,000-query budget; cached repeats are free; clean screening is reported separately.
* Batched queries (HSJA / QEBA gradient estimation) are charged in full; `first_success` is the exact
  index of the first constrained success in query order (Eq. 17), and in first-success mode the
  archive is frozen there, so the reported candidate is that first success. Images sent after it in
  the same batch are reported as `queries_after_success`.
* The scheduler batches pending queries from many concurrent attacks into one detector call. This
  does not change any per-attack query count; it only improves GPU throughput. Every call is zero-padded to
  a fixed batch shape (also in clean screening), so a label never depends on which attacks shared it.
* Baselines start from the same reference pool, order and bisection as SAGE; their full parameter
  dictionaries are `sage/baselines.py:BASELINE_PARAMS`. QEBA is the spatial-resize variant QEBA-S.

## Detectors (deviation from the manuscript: DIRE replaced by AEROBLADE)

* **DIRE is not used.** The released `lsun_adm.pth` was trained on DIRE maps that the official
  `compute_dire.py` saved under the input file name, so real LSUN maps are JPEG files and generated maps
  PNG files. The classifier separates the two formats: JPEG-compressing the official PNG DIRE maps of
  SD-v1 / SD-v2 / ADM images (q95) turns its fake rate from 100% into 0-3%, while every in-memory
  (never-JPEG) map is called fake, including 99.5% of real test images. The adapter (`sage/detectors.py:DIRE`)
  is kept for reference only.
* **AEROBLADE** (Ricker et al., CVPR 2024, cited as [4]) replaces it: training-free, minimum over the SD1 VAE,
  SD2 VAE (`stabilityai/sd-vae-ft-mse`, identical weights to the SD2-base VAE) and Kandinsky 2.1 MoVQ of the
  LPIPS-VGG layer-2 distance between an image and its reconstruction (official repository @ 50fc2b2,
  weights pinned by revision in the config). It has no decision threshold, so one is frozen on the
  development split (`experiments/calibrate_threshold.py`, max balanced accuracy) before screening.
  The forward pass runs without cuDNN and with classic attention so that each label depends on the image
  alone (verified bit-identical across batch positions).
* UFD and the 256x256 bicubic canvas follow the manuscript unchanged (official 0.5 threshold). On this
  canvas UFD detects very few generated images (clean screening: 3 of 400 eligible); measured locally,
  bicubic downsampling lowers UFD's AUC from 0.93 / 0.86 (native center crop) to 0.70 / 0.51 (SD-v1 / SD-v2).
