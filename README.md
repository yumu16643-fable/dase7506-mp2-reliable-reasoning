# Selecting Reliable LLM Reasoning - DASE7506 MP2

## Follow-up optimization experiment (this branch)

This branch records a CPU-only ranking/pooling follow-up. Validation improves from 92.23% to 94.24%, but the newly frozen method's single final test evaluation is **80.08565310492506% (374/467)**. It is a negative result, not a claimed leaderboard improvement. The existing submission [#340](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/340) and `main` branch remain unchanged.

Read [the eight-page complete report](reports/optimization-report.pdf) and [the Chinese optimization summary](OPTIMIZATION_ZH.md). Reproduce the follow-up from `code/` with `python evaluate_v2.py --split test`; the matching assets are `runs/model_bundle_v2/`. Training/search are in `train_v2.py`, and the training-only three-fold diagnostic is in `cross_validate_v2.py`. All new development uses train/validation only; final test scoring occurs after the recorded freeze.

The following sections describe the original submitted baseline and its original reproduction commands.

Frozen model: `lr_context_0.01`. Official test accuracy: **81.37044967880085% (380/467)**. Validation: **92.23057644110276% (736/798)**. This is a completed local artifact; it does not claim that a leaderboard issue has been submitted.

The method concatenates 1,024 supplied embedding features, their centered counterparts and 46 reference-free text/answer statistics. A training-standardized L2 logistic model selects the largest score among parseable candidates. No question IDs, lookup tables, cached choices, external weights, generation or reference IO are used during inference. `reports/report.pdf` is the six-page report.

## Reproduce the checkpoint result without retraining

Use Python 3.12. Open a terminal in this repository. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements.txt
cd code
..\.venv\Scripts\python.exe download_data.py
..\.venv\Scripts\python.exe check_data.py
..\.venv\Scripts\python.exe evaluate.py --split validation
..\.venv\Scripts\python.exe evaluate.py --split test
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
cd code
../.venv/bin/python download_data.py
../.venv/bin/python check_data.py
../.venv/bin/python evaluate.py --split validation
../.venv/bin/python evaluate.py --split test
```

The self-contained package already includes the data; `download_data.py` verifies and skips existing valid files. The small publication package excludes the supplied data and fetches it from the immutable upstream commit `bb801f4f2e7629d270df765ab12c333b2dc616bb`. Download/install are setup steps; prediction and scoring run offline after setup. The `.gitattributes` file prevents Windows Git newline conversion from corrupting official data hashes.

`evaluate.py` starts the unmodified official prediction and independent scoring programs as two fresh processes and externally measures their combined wall time. It sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 4 before either process starts. Results are in `code/runs/test-score.json` and `test-resources.json`.

Direct official commands, from `code/` (PowerShell):

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
python predict.py --method submission.py --checkpoint runs/model_bundle --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

For an instructor audit bank, use the same frozen method with the official runner:

```bash
python predict.py --method submission.py --checkpoint runs/model_bundle --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

Apply the four thread environment settings above to the direct test and audit commands.

## Training reproduction

From `code/`:

```bash
python -m pip install -r requirements-training.txt
python train.py --output runs/retrained
```

This fits training-only scaling/weights, runs the full validation search and saves a new validation-selected configuration. It never loads test inputs or references. Do not replace the submitted frozen bundle with a new checkpoint after examining test performance. Seeds are 2026 and 2027. Training may have small numeric differences across BLAS/platforms; checkpoint evaluation avoids retraining.

The original training command was `python train.py --output runs/search`. `freeze.py` then copied the validation winner and hashed the matching source/assets before official test scoring. It should not be rerun just to reproduce the frozen result. The original freeze manifest is retained unchanged.

## Artifacts and measurements

- `code/submission.py`, `features.py`: NumPy-only inference and reference-free features.
- `code/train.py`: complete search/training process, including worse configurations.
- `code/runs/model_bundle/`: all required inference assets, **26,397 bytes total** (uncompressed NPZ plus JSON); 16 MiB allowed.
- `code/runs/freeze.json`: matching source/asset hashes, timestamp, upstream version.
- `code/runs/search/experiments.json`, `code/runs/training.log`: actual full development records; **71.97 s** total search locally.
- `code/runs/test-score.json`: unchanged official scorer output, **380/467**.
- `code/runs/test-resources.json`: **0.925 s** combined fresh-process prediction/scoring locally with four threads. No reference-Xeon measurement is claimed.
- `code/verify_inference.py`: renamed IDs, different bank size, candidate permutations and integer output checks.
- `code/analyze.py`: frozen validation error/confidence analysis; never changes the model.
- `code/benchmark_ablations.py`: validation-only fresh-process cost comparison.
- `reports/report.pdf`, `report.md`: six-page report and editable text.
- `reports/validation_choices.csv`, `validation_errors.json`: detailed validation diagnostics.
- `reports/ablation_resources.json`: measured resource comparison.

Local hardware/software: Intel Core i7-13650HX (14 cores, 20 logical processors), Windows 11, Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.1, joblib 1.4.2, threadpoolctl 3.5.0. No GPU. Evaluation processes use at most four threads. The reference Xeon machine determines official runtime eligibility; local figures include startup/imports/loading but do not flush the OS disk cache.

## Main findings and limitations

Majority voting gives 86.22% validation accuracy; statistics-only and embedding-only LR each give 89.72%; combined LR gives 92.11%, and the submitted centered LR gives 92.23%. The last difference is only one question. MLPs/ensembles do not improve the validation score. With a linear scorer, group-centering terms cancel within each question; the extra block changes fitting/regularization rather than providing nonlinear comparison at inference.

Easy consensus groups inflate validation accuracy: on the 355 validation questions with at least two distinct valid parsed answers, the frozen model gets 82.54%, versus majority's 69.01%. This is more comparable in difficulty to the test bank (81.37%). Confidence analysis is retrospective, uncalibrated and includes the same validation bank used for model selection. The method does not verify every arithmetic step or resolve all wording ambiguities. Full analysis is in the report.

## AI/reuse disclosure

OpenAI Codex provided substantive assistance with architecture/features, implementation, debugging, local experiment execution, analysis and report drafting. This is substantial assistance beyond autocomplete. The submitter should read and understand the code and evidence before submission.

The official loader, numeric parser, prediction runner, scorer and data checker are reused unchanged. Data derives from GSM8K (MIT); the supplied `code/data/LICENSE.txt` and `NOTICE.txt` are retained. Candidate traces/representations were supplied using Qwen3-0.6B, with model revision in `data/manifest.json`. No extra pretrained weights, external datasets or other solution implementations are used. scikit-learn 1.5.1 supplies the training algorithms.

## Submission

The course requires public code/report and checkpoint/configuration links, then an actual GitHub issue created via the [MP2 submission website](https://xudongwu-0.github.io/courses/dase7506/mp2/). A local ZIP or prepared issue link is not a completed submission. Publish this repository, identify the matching commit, and link the `code/runs/model_bundle/` folder at that commit. See `SUBMISSION_ZH.md` for the concrete student ID, exact score and remaining steps.
