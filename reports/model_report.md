# Model Report

Backend: `local` | target: `default_next_month`

## Test set performance

| flavour | algorithm | ROC-AUC | PR-AUC | Brier | precision | recall | F1 | cost | savings vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| sklearn | hist_gradient_boosting | 0.7894 | 0.5377 | 0.1372 | 0.3692 | 0.7882 | 0.5028 | 8,741,000 | 27.0% |
| torch | neural network | 0.7846 | 0.5383 | 0.1367 | 0.3627 | 0.8020 | 0.4995 | 8,727,000 | 27.1% |

### sklearn

- operating threshold: **0.1818** (chosen on validation, cost-minimising)
- confusion matrix: TN=2,118 FP=1,367 FN=215 TP=800
- cost per client: NT$1,942
- baselines: intervene-nobody NT$16,240,000, intervene-everybody NT$11,977,500

### torch

- operating threshold: **0.1786** (chosen on validation, cost-minimising)
- confusion matrix: TN=2,055 FP=1,430 FN=201 TP=814
- cost per client: NT$1,939
- baselines: intervene-nobody NT$16,240,000, intervene-everybody NT$11,977,500

## Fairness audit (protected attribute excluded from features)

| group   |    n |   actual_default_rate |   selection_rate |   recall |   precision |   roc_auc |
|:--------|-----:|----------------------:|-----------------:|---------:|------------:|----------:|
| male    | 1829 |                0.2564 |           0.5068 |   0.791  |      0.4002 |    0.7834 |
| female  | 2671 |                0.2044 |           0.4642 |   0.7857 |      0.346  |    0.7924 |

## Calibration (test, by decile)

|   bin |   n |   mean_predicted |   observed_rate |     gap |
|------:|----:|-----------------:|----------------:|--------:|
|     0 | 450 |           0.0248 |          0.04   | -0.0152 |
|     1 | 450 |           0.0688 |          0.0556 |  0.0133 |
|     2 | 450 |           0.1017 |          0.0822 |  0.0195 |
|     3 | 450 |           0.1047 |          0.1089 | -0.0042 |
|     4 | 450 |           0.1367 |          0.1667 | -0.03   |
|     5 | 450 |           0.1813 |          0.1489 |  0.0324 |
|     6 | 450 |           0.2083 |          0.2356 | -0.0273 |
|     7 | 450 |           0.2479 |          0.2911 | -0.0433 |
|     8 | 450 |           0.4336 |          0.4067 |  0.0269 |
|     9 | 450 |           0.72   |          0.72   | -0      |

## Cost sensitivity

Operating point as the false-negative : false-positive cost ratio varies.

|   fn_fp_ratio |   threshold |   flagged_pct |   recall |   precision |   savings_pct |
|--------------:|------------:|--------------:|---------:|------------:|--------------:|
|             2 |      0.5214 |          12.4 |   0.3714 |      0.6732 |          18.8 |
|             3 |      0.2596 |          27.6 |   0.6069 |      0.4964 |          30   |
|             5 |      0.2069 |          40.6 |   0.7409 |      0.4116 |          30.7 |
|             8 |      0.1064 |          62.7 |   0.8966 |      0.3227 |          19.7 |
|            10 |      0.1064 |          62.7 |   0.8966 |      0.3227 |          14.5 |
|            15 |      0.1017 |          83   |   0.9704 |      0.2636 |           7.5 |
|            20 |      0.1017 |          83   |   0.9704 |      0.2636 |           3.7 |
|            30 |      0      |         100   |   1      |      0.2256 |           0   |
