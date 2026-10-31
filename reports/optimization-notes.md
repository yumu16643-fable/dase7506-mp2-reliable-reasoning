6. Follow-up: optimizing the ranking objective

The original submitted verifier and result on pages 1-6 remain a separately frozen artifact. This follow-up tests a new hypothesis: directly optimizing within-question choice, rather than independent candidate correctness, may better match selection accuracy. It is reported in full even though its final test result does not improve.

Protocol and training objective

The follow-up uses only supplied training inputs/references and validation development. The original test score was already public; no test inputs or references enter the follow-up training, search or three-fold diagnostic. The new method is selected on validation and frozen before its one final test evaluation. No parameters are changed afterward.

Input features are the 1,024 frozen embedding dimensions plus the original 46 statistics (1,070 total); the centered duplicate block is omitted. Scaling is fitted on training candidates only. Training focuses on the 2,114 training groups with at least one valid wrong candidate. Groups with no valid wrong answer need no learned discrimination after invalid-answer masking. No IDs or stored choices are features.

The selected listwise loss is the mean of logsumexp over valid candidate logits minus logsumexp over correct candidate logits, plus 0.1 times squared weight norm divided by two. This rewards probability assigned to any correct candidate. A linear model is optimized by L-BFGS-B. The selected fit converges in 28 iterations; weights are exported as float32 for NumPy-only inference.

For candidates sharing a parsed answer, pooled score is T * log(mean(exp(z/T))), with T = 0.5. The highest-scoring answer group wins. Log-mean pooling avoids automatically multiplying evidence by the number of duplicate answers. Invalid parsed candidates remain masked.

[['Validation comparison', 'Accuracy %'], ['Original submitted centered BCE LR', '92.23'], ['BCE LR with combined features', '92.11'], ['Best pairwise combined LR', '92.73'], ['Best uniform-positive listwise combined model', '93.48'], ['Selected marginal listwise model, no pooling', '93.86'], ['Selected marginal model with answer pooling', '94.24']]

The bounded search fits 20 models: eight pairwise logistic models and twelve linear listwise models. It then compares ten fixed pooling settings per model plus the original baseline, for 210 validation comparisons. The winner solves 752/798 validation questions, 16 more than the original 736/798. The no-pooling comparison isolates the final pooling step; other comparisons change the objective and training-group treatment together.

7. Follow-up evidence, costs and negative result

[['Evaluation', 'Original verifier', 'Follow-up verifier'], ['Validation', '736/798 (92.23%)', '752/798 (94.24%)'], ['Informative validation subset', '293/355 (82.54%)', '309/355 (87.04%)'], ['Training-only 3-fold held-out groups', '4318/4755 (90.81%)', '4329/4755 (91.04%)'], ['Final official test', '380/467 (81.37%)', '374/467 (80.09%)']]

<b>The final test improvement did not materialize.</b> The follow-up loses six test questions (-1.28 percentage points), despite the validation gain. The already-submitted baseline is left unchanged. The follow-up weights, freeze record, predictions and independent scorer JSON are retained as a transparent failed experiment, not presented as an improved leaderboard result.

Reliability and interpretation

On validation, the follow-up fixes 29 original mistakes and introduces 13. The paired 5,000-resample question bootstrap gives a +0.50 to +3.63 percentage-point gain interval. This interval is descriptive and does not correct for selecting among 210 validation comparisons. The much smaller training-only three-fold gain (+0.23 points) provides a caution against treating the +2.01-point validation gain as robust. The three-fold configuration is itself chosen after validation search, so this is a robustness diagnostic, not an untouched confirmatory study.

Possible explanations include development-set selection effects, split difficulty and sampling variation. The available results do not identify a single cause. No test-error-driven feature changes or second test-based search were performed. The finding supports caution about replacing a simple verifier solely because a more task-aligned loss looks better on one validation bank.

Resources, reproduction and checks

CPU only: Intel Core i7-13650HX, no GPU. Follow-up development takes 51.45 s; the three-fold diagnostic adds 18.06 s. The frozen inference bundle contains 14,214 bytes, versus the original 26,397 bytes. Complete fresh-process test prediction plus scoring takes 0.928 s locally with four threads. These are local measurements; the reference Xeon machine was not tested.

From code/, install requirements-training.txt and run train_v2.py to reproduce the search, or cross_validate_v2.py for the training-only diagnostic. For checkpoint reproduction, install requirements.txt and run evaluate_v2.py --split validation or --split test. The unchanged official runner/scorer are used. Inference assets are in runs/model_bundle_v2; runs/freeze-v2.json records matching source/asset hashes before final scoring. The finite-difference gradient error is 2.01e-8; float32 export preserves every validation choice; renamed IDs, a seven-question bank and candidate permutation checks pass.

AI assistance and source attribution on page 6 also apply to this follow-up. Codex provided substantial design, implementation, execution, analysis and writing assistance. This follow-up adds no external data or pretrained weights and uses the same pinned dependencies. Frozen source and weights remain unchanged after the reported test.
