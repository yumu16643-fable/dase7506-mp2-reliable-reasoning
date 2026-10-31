# DASE7506 MP2 - Frozen arithmetic verifier

**Current final test accuracy: 82.44111349036403% (385/467).** Validation: 94.23558897243107% (752/798). This is +5 test questions / +1.07066381 percentage points over the original 81.37044967880085% submission.

The submitted method is **`code/submission_arithmetic.py`**, with checkpoint **`code/runs/model_bundle_arithmetic/`**. The report for this version is **[reports/arithmetic-report.pdf](reports/arithmetic-report.pdf)** (four pages). Other method files and reports are historical experiments; do not substitute their checkpoints or default commands when reproducing the current score.

## Checkpoint reproduction, no training required

Use Python 3.12. Windows PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements.txt
cd code
..\.venv\Scripts\python.exe download_data.py
..\.venv\Scripts\python.exe check_data.py
..\.venv\Scripts\python.exe check_arithmetic.py
..\.venv\Scripts\python.exe evaluate_arithmetic.py
..\.venv\Scripts\python.exe evaluate_arithmetic_final.py
```

On Linux/macOS, use `.venv/bin/python` from the root or `../.venv/bin/python` after entering `code/`. The wrappers start fresh official prediction and independent scoring processes and set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 4 before either starts. The final wrapper verifies the original frozen source/asset hashes; it does not retrain or select a model.

Equivalent direct official test commands, from `code/` (PowerShell):

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
python predict.py --method submission_arithmetic.py --checkpoint runs/model_bundle_arithmetic --split test --output runs/arithmetic-test-predictions.json
python score_predictions.py --split test --predictions runs/arithmetic-test-predictions.json --output runs/arithmetic-test-score.json
```

`download_data.py` obtains the official bank from upstream commit `bb801f4f2e7629d270df765ab12c333b2dc616bb` and verifies its manifest hashes. Download/install are setup only. Prediction and scoring run offline afterward. `.gitattributes` prevents newline conversion from invalidating data/source hashes. An instructor audit bank is accepted without retraining or ID assumptions:

```bash
python predict.py --method submission_arithmetic.py --checkpoint runs/model_bundle_arithmetic --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

Apply the same four thread settings to direct audit evaluation. Inference uses the supplied bank/arrays and never opens reference files, saved choices or answer lookups.

## Method and model selection

The input consists of 1,024 supplied embedding dimensions, 46 original trace/answer statistics, and 18 deterministic arithmetic-consistency features (1,088 total). A bounded AST interpreter checks supported numerical equations, fractions, percentages and LaTex; unknown/unit-bearing expressions can be skipped. It never calls Python `eval` or executes expressions as code. The features are evidence, not hard vetoes or a full mathematical proof.

A standardized linear model optimizes marginal listwise loss, rewarding probability on any correct valid candidate, with L2=0.1. Normalization and weights fit training only. The configuration is selected by training-only three-fold question-disjoint OOF accuracy, seed 2026; the original baseline wins ties. A diagnostic reshuffle uses seed 2037 without reselection. Validation is inspected after the selected configuration is fixed, not used for a new parameter search.

| Evaluation | Original model | Arithmetic verifier |
|---|---:|---:|
| Train OOF, seed 2026 | 90.81% (4318/4755) | 91.65% (4358/4755) |
| Reshuffled OOF, seed 2037 | 90.75% | 91.84% |
| Validation | 92.23% (736/798) | 94.24% (752/798) |
| Official public test | 81.37% (380/467) | 82.44% (385/467) |

Matched objective/regularization ablation: 91.23% OOF without arithmetic versus 91.65% with it. Development does not load test inputs/references. The original freeze is recorded before the separately authorized final test evaluation; no model/features/settings are changed after its score. Earlier project test results (including the CPU follow-up's 80.09%) are preserved. Repeated project use of the development/public banks limits independent-generalization conclusions; the instructor audit remains important.

## Resources and checks

- Required inference assets: **14,323 bytes total**, uncompressed NPZ and configuration, excluding supplied data and ordinary source.
- Local complete test prediction plus scoring: **1.4095039 s**, two fresh processes, four threads, startup/imports/hashing/loading/preprocessing included.
- Hardware: Intel Core i7-13650HX; Python 3.12.7; CPU training and inference for this version. No GPU required.
- Reference Xeon runtime is not measured. The OS disk cache is not flushed for local timing.
- Corrected full training search: 131.98 s; reshuffled diagnostic: 56.73 s; matched ablation: 19.26 s. Preliminary parser-debug run was aborted before validation; its partial log is retained.
- Numeric/LaTex/rounding/unit/unsafe-expression behavioral checks, float32 export choices, renamed IDs, different bank size and candidate permutation checks pass.

Key files:

- `code/runs/model_bundle_arithmetic/`: all required inference assets.
- `code/runs/freeze-arithmetic.json`: original model/source hashes and development freeze.
- `code/runs/arithmetic-final-evaluation.json`: final-scoring scope and hash verification after development.
- `code/runs/arithmetic-test-score.json`: unchanged official scorer output, 385/467.
- `code/runs/arithmetic-test-resources.json`: full fresh-process measurement.
- `code/runs/arithmetic-search/`: model comparison, reshuffle, ablation and parser audit.
- `reports/arithmetic-report.pdf`: current four-page report; other reports are historical.
- `SHA256SUMS.txt`: all published package-file hashes.

## Training reproduction

From `code/`, in a separate environment/directory so the frozen bundle is not overwritten:

```bash
python -m pip install -r requirements-training.txt
python train_arithmetic.py
python check_robustness_arithmetic.py
python ablate_arithmetic.py
```

These scripts never open test inputs or references. Exact dependencies: NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.1, joblib 1.4.2 and threadpoolctl 3.5.0. Inference requires NumPy only. Learned scaling/statistics are included in the checkpoint, so evaluation needs no retraining. Small cross-platform training differences are possible; checkpoint scoring avoids retraining.

## AI assistance and attribution

OpenAI Codex provided substantive assistance with design, features, implementation, debugging, experiment execution, analysis and report writing, beyond autocomplete. The submitter should understand the method and evidence. Official bank loading, numeric parsing, runner, data checker and independent scorer are reused unchanged. No extra pretrained weights, external datasets or cached-answer selection are used.

Questions/references derive from GSM8K (retained MIT license and attribution in `DATA_LICENSE.txt`, `DATA_NOTICE.txt`, and downloaded data). Candidate traces/embeddings were supplied using Qwen3-0.6B; revision is recorded in the data manifest. Upstream course repository: https://github.com/xudongwu-0/DASE7506-MP2 .

Historical development documentation is in `README_HISTORY.md` and the separate experiment summaries. Original submission #340 remains preserved; the final score update uses this version and its matching checkpoint links.
