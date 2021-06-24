x-vector-procrustes
===================

### Installation

```sh
./install.sh
```

### Experiments

### VoicePrivacy

VoicePrivacy linkability Scores *without* retrained ASV on anon data:
```sh
./run.sh --stage 0 --skip-stage "1 2 3 4"
```
<details>

<summary> results:</summary>

```
**ASV (original): test_trials_f original <=> test_enrolls - original**
EER: 10.33%
**ASV (original): test_trials_m original <=> test_enrolls - original**
EER: 2.85%
**ASV (original): test_trials_f anonymized <=> test_enrolls - original**
EER: 48.99%
**ASV (original): test_trials_m anonymized <=> test_enrolls - original**
EER: 42.64%
**ASV (original): test_trials_f anonymized <=> test_enrolls - anonymized**
EER: 29.41%
**ASV (original): test_trials_m anonymized <=> test_enrolls - anonymized**
EER: 29.15%
```

</details>

VoicePrivacy linkability Scores *with* retrained ASV on anon data:
```sh
./run.sh --stage 0 --skip-stage "1 2 3 4" --retrained-anon-xtractor true
```

<details>

<summary> results:</summary>

```
Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV (anon): test_trials_f anonymized <=> test_enrolls - anonymized**
EER: 17.12%
**ASV (anon): test_trials_m anonymized <=> test_enrolls - anonymized**
EER: 14.08%
```

</details>


---

### Procrustes
#### x-tractor original


Procrustes linkability and reversibility Scores x-tractor original:
```sh
./run.sh --stage 1
```

<details>

<summary> results:</summary>

```
== Training rotation ==
Compute done, rotation shape : (512, 512)
Top   1:        96.12    34.47
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        96.12    34.47
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:        25.48    1.36
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:        36.61    2.36
=== TEST likability between Anonymized and Orignal speech ===
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 41.94%
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 30.55%
```

</details>

Procrustes linkability and reversibility Scores x-tractor original gender specific training:
```sh
./run.sh --stage 1 --filter_gender f
./run.sh --stage 1 --filter_gender m
```

<details>

<summary> results (merged):</summary>

```
 == Data used to train procrustes uv ==
Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Compute done, rotation shape : (512, 512)
Top   1:        96.46    47.64
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        59.59    27.85
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:        26.57    1.77
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 40.12%


 == Data used to train procrustes uv ==
Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Compute done, rotation shape : (512, 512)
Top   1:        97.28    54.35
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        47.26    23.29
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:        45.80    2.36
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 30.96%
```

</details>


Procrustes linkability and reversibility Scores x-tractor original with pca on x-vector:
```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results:</summary>

```
== Training rotation ==
Computing PCA, 70 dimensions
(438, 70) total explained variance ratio : 0.98974127
(438, 70) total explained variance ratio : 0.98620903
Compute done, rotation shape : (70, 70)
Top   1:        91.10    22.60
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Loading pca from: exp/enroll_train_wp
Top   1:        91.10    22.60
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:        27.11    1.77
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:        40.81    2.49
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 32.63%
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 32.74%
```

</details>

Procrustes linkability and reversibility Scores x-tractor original gender specific training with pca on x-vector:
```sh
./run.sh --stage 1 --filter_gender f --frontend-train "--pca --pca_n_dim 70"
./run.sh --stage 1 --filter_gender m --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results (merged):</summary>

```
 == Data used to train procrustes uv ==
Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Computing PCA, 70 dimensions
(254, 70) total explained variance ratio : 0.99149513
(254, 70) total explained variance ratio : 0.97984266
Compute done, rotation shape : (70, 70)
Top   1:        94.49    40.94
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Loading pca from: exp/enroll_train_wp
Top   1:        55.25    23.52
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:        30.52    2.04
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 25.35%


 == Data used to train procrustes uv ==
Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Computing PCA, 70 dimensions
(184, 70) total explained variance ratio : 0.9933938
(184, 70) total explained variance ratio : 0.9853472
Compute done, rotation shape : (70, 70)
Top   1:        97.83    48.37
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:        50.00    3.94
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 24.45%
```

</details>


---

### Procrustes
#### x-tractor anon

Procrustes linkability and reversibility Scores x-tractor anon:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true
```

<details>

<summary> results:</summary>

```
== Training rotation ==
Compute done, rotation shape : (512, 512)
Top   1:        97.95    30.59
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        97.95    30.59
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:        58.72    3.27
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:        59.19    2.89
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 28.99%
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 23.65%
```

</details>

Procrustes linkability and reversibility Scores x-tractor anon gender specific training:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender f
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender m
```

<details>

<summary> results (merged):</summary>

```
 == Data used to train procrustes uv ==
Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Compute done, rotation shape : (512, 512)
Top   1:        99.21    53.54
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        63.93    31.51
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:        54.77    3.27
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 27.01%


 == Data used to train procrustes uv ==
Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Compute done, rotation shape : (512, 512)
Top   1:        99.46    53.26
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        46.58    23.06
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:        56.56    3.02
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 21.15%
```

</details>

Procrustes linkability and reversibility Scores x-tractor anon with pca on x-vector:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results:</summary>

```
== Training rotation ==
Computing PCA, 70 dimensions
(438, 70) total explained variance ratio : 0.98974365
(438, 70) total explained variance ratio : 0.98574036
Compute done, rotation shape : (70, 70)
Top   1:        95.66    21.92
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Loading pca from: exp/enroll_train_wp
Top   1:        95.66    21.92
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:        51.91    2.59
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:        56.96    2.36
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 21.54%
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 23.14%
```

</details>



Procrustes linkability and reversibility Scores x-tractor anon gender specific training and pca on x-vector:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender f --frontend-train "--pca --pca_n_dim 70"
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender m --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results (merged):</summary>

```
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Loading pca from: exp/enroll_train_wp
Top   1:        57.53    24.66
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:        59.81    3.27
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 14.60%


== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Loading pca from: exp/enroll_train_wp
Top   1:        42.01    21.69
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:        59.97    3.15
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 13.11%
```

</details>

---

### Wasserstein Procrustes
#### x-tractor original

Wasserstein Procrustes linkability and reversibility Scores x-tractor original:
```sh
./run.sh --stage 1 --wp true
```

<details>

<summary> results:</summary>

```
Alignment Done
Compute done, rotation shape : (512, 512)
Top   1:        92.92    16.44
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Top   1:        92.92    16.44
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:        25.20    1.36
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:        22.05    1.57
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 43.56%
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 33.38%
```

</details>


Wasserstein Procrustes linkability and reversibility Scores x-tractor original gender specific training:
```sh
./run.sh --stage 1 --filter_gender f --wp true
./run.sh --stage 1 --filter_gender m --wp true
```

<details>

<summary> results (merged):</summary>

```
Filtering by gender f
Alignment Done
Compute done, rotation shape : (512, 512)
Top   1:        94.49    16.54
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:        26.57    2.04
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 40.70%


Filtering by gender m
Alignment Done
Compute done, rotation shape : (512, 512)
Top   1:        96.74    18.48
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:        20.87    0.39
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 35.92%
```

</details>

Wasserstein Procrustes linkability and reversibility Scores x-tractor original with pca on x-vector:
```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --wp true
```

<details>

<summary> results:</summary>

```
== Training rotation ==
Computing PCA, 70 dimensions
(438, 70) total explained variance ratio : 0.98973745
(438, 70) total explained variance ratio : 0.9861987
Alignment Done
Compute done, rotation shape : (70, 70)
Top   1:        93.38    12.33
Done
== TEST rotation irreversibility ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Loading pca from: exp/enroll_train_wp
Top   1:        93.38    12.33
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:        24.11    1.23
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:        38.58    1.84
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 36.30%
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 35.36%
```

</details>


Wasserstein Procrustes linkability and reversibility Scores x-tractor original gender specific training with pca on x-vector:
```sh
./run.sh --stage 1 --filter_gender f --frontend-train "--pca --pca_n_dim 70" --wp true
./run.sh --stage 1 --filter_gender m --frontend-train "--pca --pca_n_dim 70" --wp true
```

<details>

<summary> results (merged):</summary>

```md
Filtering by gender f
== Training rotation ==
Computing PCA, 70 dimensions
(254, 70) total explained variance ratio : 0.9914975
(254, 70) total explained variance ratio : 0.9798415
Alignment Done
Compute done, rotation shape : (70, 70)
Top   1:        92.52    18.90
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:        16.76    1.09
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 26.67%


Filtering by gender m
== Training rotation ==
Computing PCA, 70 dimensions
(184, 70) total explained variance ratio : 0.9933953
(184, 70) total explained variance ratio : 0.9853457
Alignment Done
Compute done, rotation shape : (70, 70)
Top   1:        95.11    23.37
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:        45.28    1.57
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
EER: 27.17%
```

</details>


---
