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

<summary> results (Clickable :computer_mouse:):</summary>

```wiki
Reproduce VoicePrivacy EER results with cosine scoring
**ASV (original): test_trials_f original <=> test_enrolls - original**
ROC_EER: 10.33
EER: 10.13
Cllr (min/act): 0.354056 0.872250
linkability: 0.768135
**ASV (original): test_trials_m original <=> test_enrolls - original**
ROC_EER: 2.85
EER: 2.72
Cllr (min/act): 0.087939 0.838062
linkability: 0.910283
**ASV (original): test_trials_f anonymized <=> test_enrolls - original**
ROC_EER: 48.99
EER: 49.36
Cllr (min/act): 0.999199 1.022628
linkability: 0.068093
**ASV (original): test_trials_m anonymized <=> test_enrolls - original**
ROC_EER: 42.64
EER: 49.29
Cllr (min/act): 0.996365 1.033778
linkability: 0.148108
**ASV (original): test_trials_f anonymized <=> test_enrolls - anonymized**
ROC_EER: 29.41
EER: 29.03
Cllr (min/act): 0.801984 1.106415
linkability: 0.327953
**ASV (original): test_trials_m anonymized <=> test_enrolls - anonymized**
ROC_EER: 29.15
EER: 28.87
Cllr (min/act): 0.816143 1.110715
linkability: 0.303574

Spk verif scores:
ASV-libri_test_enrolls_anon-libri_test_trials_f_anon
  EER: 31.39%
  Cllr (min/act): 0.830/15.787
  ROCCH-EER: 30.943%
ASV-libri_test_enrolls_anon-libri_test_trials_m_anon
  EER: 36.75%
  Cllr (min/act): 0.909/34.970
  ROCCH-EER: 36.281%
with retrained x-vector:
ASV-libri_test_enrolls_anon-libri_test_trials_f_anon
  EER: 31.39%
  Cllr (min/act): 0.830/15.787
  ROCCH-EER: 30.943%
ASV-libri_test_enrolls_anon-libri_test_trials_m_anon
  EER: 36.75%
  Cllr (min/act): 0.909/34.970
  ROCCH-EER: 36.281%
---
```

</details>

VoicePrivacy linkability Scores *with* retrained ASV on anon data:
```sh
./run.sh --stage 0 --skip-stage "1 2 3 4" --retrained-anon-xtractor true
```

<details>

<summary> results:</summary>

```wiki
Reproduce VoicePrivacy EER results with cosine scoring
Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV (anon): test_trials_f anonymized <=> test_enrolls - anonymized**
ROC_EER: 17.12
EER: 16.90
Cllr (min/act): 0.513022 1.076549
linkability: 0.575353
**ASV (anon): test_trials_m anonymized <=> test_enrolls - anonymized**
ROC_EER: 14.08
EER: 13.59
Cllr (min/act): 0.487963 1.084177
linkability: 0.608591

Spk verif scores:
ASV-libri_test_enrolls_anon-libri_test_trials_f_anon
  EER: 31.39%
  Cllr (min/act): 0.830/15.787
  ROCCH-EER: 30.943%
ASV-libri_test_enrolls_anon-libri_test_trials_m_anon
  EER: 36.75%
  Cllr (min/act): 0.909/34.970
  ROCCH-EER: 36.281%
with retrained x-vector:
ASV-libri_test_enrolls_anon-libri_test_trials_f_anon
  EER: 11.5%
  Cllr (min/act): 0.364/2.832
  ROCCH-EER: 11.278%
ASV-libri_test_enrolls_anon-libri_test_trials_m_anon
  EER: 9.354%
  Cllr (min/act): 0.305/3.941
  ROCCH-EER: 9.171%
---
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

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	96.12 (speaker accuracy)	 34.47 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	25.48 (speaker accuracy)	 1.36 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	36.61 (speaker accuracy)	 2.36 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 41.94
EER: 39.37
Cllr (min/act): 0.889367 1.009684
linkability: 0.199870
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 30.55
EER: 30.47
Cllr (min/act): 0.834911 0.999177
linkability: 0.310689
```

</details>

Procrustes linkability and reversibility Scores x-tractor original gender specific training:
```sh
./run.sh --stage 1 --filter_gender f
./run.sh --stage 1 --filter_gender m
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	96.46 (speaker accuracy)	 47.64 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	26.57 (speaker accuracy)	 1.77 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 40.12
EER: 37.58
Cllr (min/act): 0.875361 1.009219
linkability: 0.218478


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	97.28 (speaker accuracy)	 54.35 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	45.80 (speaker accuracy)	 2.36 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 30.96
EER: 30.75
Cllr (min/act): 0.825736 1.003011
linkability: 0.288890
```

</details>


Procrustes linkability and reversibility Scores x-tractor original with pca on x-vector:
```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results:</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (438, 70) with a total explained variance ratio on clear data: 0.98974174
Output shape after PCA: (438, 70) total explained variance ratio on target(anonymized) data: 0.98620886
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	91.32 (speaker accuracy)	 23.06 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	27.52 (speaker accuracy)	 1.77 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	40.29 (speaker accuracy)	 2.49 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 32.64
EER: 32.32
Cllr (min/act): 0.802954 0.962672
linkability: 0.275204
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 32.72
EER: 32.43
Cllr (min/act): 0.848538 0.998714
linkability: 0.278785
```

</details>

Procrustes linkability and reversibility Scores x-tractor original gender specific training with pca on x-vector:
```sh
./run.sh --stage 1 --filter_gender f --frontend-train "--pca --pca_n_dim 70"
./run.sh --stage 1 --filter_gender m --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (254, 70) with a total explained variance ratio on clear data: 0.9914968
Output shape after PCA: (254, 70) total explained variance ratio on target(anonymized) data: 0.9798316
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	94.49 (speaker accuracy)	 40.16 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	30.52 (speaker accuracy)	 1.91 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 25.33
EER: 24.67
Cllr (min/act): 0.708871 0.919555
linkability: 0.414599


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (184, 70) with a total explained variance ratio on clear data: 0.9933948
Output shape after PCA: (184, 70) total explained variance ratio on target(anonymized) data: 0.9853459
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	97.83 (speaker accuracy)	 48.91 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	50.39 (speaker accuracy)	 4.20 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 24.46
EER: 24.27
Cllr (min/act): 0.687435 0.905601
linkability: 0.443180
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

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	97.95 (speaker accuracy)	 30.59 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	58.72 (speaker accuracy)	 3.27 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	59.19 (speaker accuracy)	 2.89 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 28.99
EER: 28.75
Cllr (min/act): 0.756053 0.987522
linkability: 0.374424
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 23.65
EER: 23.10
Cllr (min/act): 0.663404 0.977133
linkability: 0.457706
```

</details>

Procrustes linkability and reversibility Scores x-tractor anon gender specific training:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender f
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender m
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	99.21 (speaker accuracy)	 53.54 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	54.77 (speaker accuracy)	 3.27 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 27.01
EER: 26.56
Cllr (min/act): 0.704388 0.984556
linkability: 0.425735


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	99.46 (speaker accuracy)	 53.26 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	56.56 (speaker accuracy)	 3.02 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 21.15
EER: 21.15
Cllr (min/act): 0.612003 0.976945
linkability: 0.481244
```

</details>

Procrustes linkability and reversibility Scores x-tractor anon with pca on x-vector:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results:</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (438, 70) with a total explained variance ratio on clear data: 0.9897418
Output shape after PCA: (438, 70) total explained variance ratio on target(anonymized) data: 0.98574746
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	95.43 (speaker accuracy)	 21.46 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	52.45 (speaker accuracy)	 2.86 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	57.48 (speaker accuracy)	 2.49 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 21.55
EER: 21.21
Cllr (min/act): 0.599200 0.937384
linkability: 0.491354
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 23.16
EER: 22.98
Cllr (min/act): 0.669663 0.968386
linkability: 0.456137
```

</details>



Procrustes linkability and reversibility Scores x-tractor anon gender specific training and pca on x-vector:
```sh
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender f --frontend-train "--pca --pca_n_dim 70"
./run.sh --stage 1 --retrained-anon-xtractor true --filter_gender m --frontend-train "--pca --pca_n_dim 70"
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (254, 70) with a total explained variance ratio on clear data: 0.99149674
Output shape after PCA: (254, 70) total explained variance ratio on target(anonymized) data: 0.97646797
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	99.21 (speaker accuracy)	 42.52 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	60.22 (speaker accuracy)	 3.27 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 14.62
EER: 14.48
Cllr (min/act): 0.457237 0.864921
linkability: 0.653441


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (184, 70) with a total explained variance ratio on clear data: 0.99339336
Output shape after PCA: (184, 70) total explained variance ratio on target(anonymized) data: 0.9861342
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	100.00 (speaker accuracy)	 51.09 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	59.58 (speaker accuracy)	 3.02 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 13.12
EER: 13.11
Cllr (min/act): 0.453624 0.846025
linkability: 0.648048
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

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	91.55 (speaker accuracy)	 15.53 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	26.02 (speaker accuracy)	 1.36 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	25.85 (speaker accuracy)	 1.57 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 43.96
EER: 40.78
Cllr (min/act): 0.917082 1.013459
linkability: 0.169659
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 33.90
EER: 33.32
Cllr (min/act): 0.881784 1.005720
linkability: 0.260842
```

</details>


Wasserstein Procrustes linkability and reversibility Scores x-tractor original gender specific training:
```sh
./run.sh --stage 1 --filter_gender f --wp true
./run.sh --stage 1 --filter_gender m --wp true
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	98.43 (speaker accuracy)	 20.47 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	25.61 (speaker accuracy)	 1.50 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 40.81
EER: 39.93
Cllr (min/act): 0.919463 1.014105
linkability: 0.166534


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	97.83 (speaker accuracy)	 19.02 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	21.26 (speaker accuracy)	 0.66 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 35.24
EER: 35.01
Cllr (min/act): 0.881113 1.009955
linkability: 0.226058
```

</details>

Wasserstein Procrustes linkability and reversibility Scores x-tractor original with pca on x-vector:
```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --wp true
```

<details>

<summary> results:</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (438, 70) with a total explained variance ratio on clear data: 0.9897431
Output shape after PCA: (438, 70) total explained variance ratio on target(anonymized) data: 0.9862072
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	87.44 (speaker accuracy)	 12.79 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	19.62 (speaker accuracy)	 0.95 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	42.65 (speaker accuracy)	 1.97 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 37.47
EER: 36.18
Cllr (min/act): 0.861870 0.971982
linkability: 0.217861
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 37.68
EER: 36.33
Cllr (min/act): 0.892842 1.005039
linkability: 0.216760
```

</details>


Wasserstein Procrustes linkability and reversibility Scores x-tractor original gender specific training with pca on x-vector:
```sh
./run.sh --stage 1 --filter_gender f --frontend-train "--pca --pca_n_dim 70" --wp true
./run.sh --stage 1 --filter_gender m --frontend-train "--pca --pca_n_dim 70" --wp true
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (254, 70) with a total explained variance ratio on clear data: 0.9914963
Output shape after PCA: (254, 70) total explained variance ratio on target(anonymized) data: 0.9798459
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	94.49 (speaker accuracy)	 23.23 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	26.16 (speaker accuracy)	 1.36 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 28.15
EER: 27.41
Cllr (min/act): 0.745610 0.925965
linkability: 0.369201


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (184, 70) with a total explained variance ratio on clear data: 0.993396
Output shape after PCA: (184, 70) total explained variance ratio on target(anonymized) data: 0.98534644
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	98.37 (speaker accuracy)	 25.00 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	50.52 (speaker accuracy)	 2.23 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 26.55
EER: 26.55
Cllr (min/act): 0.749897 0.915703
linkability: 0.376375
```

</details>

---

### Wasserstein Procrustes
#### x-tractor anon

Wasserstein Procrustes linkability and reversibility Scores x-tractor anon:
```sh
./run.sh --stage 1 --wp true --retrained-anon-xtractor true
```

<details>

<summary> results:</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	99.32 (speaker accuracy)	 14.38 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	56.54 (speaker accuracy)	 2.45 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	57.48 (speaker accuracy)	 2.89 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 31.35
EER: 30.79
Cllr (min/act): 0.812413 0.992908
linkability: 0.325512
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 25.62
EER: 24.64
Cllr (min/act): 0.689764 0.980588
linkability: 0.442820
```

</details>


Wasserstein Procrustes linkability and reversibility Scores x-tractor anon gender specific training:
```sh
./run.sh --stage 1 --filter_gender f --wp true --retrained-anon-xtractor true
./run.sh --stage 1 --filter_gender m --wp true --retrained-anon-xtractor true
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	99.21 (speaker accuracy)	 28.74 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Top   1:	51.50 (speaker accuracy)	 3.27 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 29.17
EER: 28.61
Cllr (min/act): 0.736770 0.988891
linkability: 0.372278


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (512, 512)
Top   1:	99.46 (speaker accuracy)	 18.48 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Top   1:	50.26 (speaker accuracy)	 2.49 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
ROC_EER: 24.81
EER: 24.00
Cllr (min/act): 0.681331 0.983677
linkability: 0.435266
```

</details>

Wasserstein Procrustes linkability and reversibility Scores x-tractor anon with pca on x-vector:
```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --wp true --retrained-anon-xtractor true
```

<details>

<summary> results:</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (438, 70) with a total explained variance ratio on clear data: 0.98974335
Output shape after PCA: (438, 70) total explained variance ratio on target(anonymized) data: 0.9857518
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	94.98 (speaker accuracy)	 14.16 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	44.82 (speaker accuracy)	 2.18 (segment accuracy)
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	58.66 (speaker accuracy)	 2.36 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 24.23
EER: 24.06
Cllr (min/act): 0.670981 0.947002
linkability: 0.431526
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 25.22
EER: 24.40
Cllr (min/act): 0.697781 0.972370
linkability: 0.418139
```

</details>


Wasserstein Procrustes linkability and reversibility Scores x-tractor anon gender specific training with pca on x-vector:
```sh
./run.sh --stage 1 --filter_gender f --frontend-train "--pca --pca_n_dim 70" --wp true --retrained-anon-xtractor true
./run.sh --stage 1 --filter_gender m --frontend-train "--pca --pca_n_dim 70" --wp true --retrained-anon-xtractor true
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender f
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 254
x_vector_u samples after filtering: 254
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (254, 70) with a total explained variance ratio on clear data: 0.99149716
Output shape after PCA: (254, 70) total explained variance ratio on target(anonymized) data: 0.9764552
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	98.43 (speaker accuracy)	 24.02 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	57.22 (speaker accuracy)	 2.86 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 15.36
EER: 14.80
Cllr (min/act): 0.468355 0.869226
linkability: 0.641547


   == Data used to train rotation ==
     - xvect_libri_test_enrolls 
     - xvect_libri_test_enrolls_anon

Filtering by gender m
x_vector_l samples: 438
x_vector_u samples: 438
x_vector_l samples after filtering: 184
x_vector_u samples after filtering: 184
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (184, 70) with a total explained variance ratio on clear data: 0.9933967
Output shape after PCA: (184, 70) total explained variance ratio on target(anonymized) data: 0.98611456
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	99.46 (speaker accuracy)	 38.59 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	59.71 (speaker accuracy)	 2.36 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 13.88
EER: 13.70
Cllr (min/act): 0.466674 0.848863
linkability: 0.638811
```

</details>


---

### Oracle Procrustes

```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --retrained-anon-xtractor true --oracle-f true
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --retrained-anon-xtractor true --oracle-m true
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_trials_f 
     - xvect_libri_test_trials_f_anon

Filtering by gender f
x_vector_l samples: 734
x_vector_u samples: 734
x_vector_l samples after filtering: 734
x_vector_u samples after filtering: 734
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (734, 70) with a total explained variance ratio on clear data: 0.9897747
Output shape after PCA: (734, 70) total explained variance ratio on target(anonymized) data: 0.97263634
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	98.77 (speaker accuracy)	 14.31 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	98.77 (speaker accuracy)	 14.31 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 12.10
EER: 12.11
Cllr (min/act): 0.444288 0.863162
linkability: 0.710707


   == Data used to train rotation ==
     - xvect_libri_test_trials_m 
     - xvect_libri_test_trials_m_anon

Filtering by gender m
x_vector_l samples: 762
x_vector_u samples: 762
x_vector_l samples after filtering: 762
x_vector_u samples after filtering: 762
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (762, 70) with a total explained variance ratio on clear data: 0.98975855
Output shape after PCA: (762, 70) total explained variance ratio on target(anonymized) data: 0.98012084
Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	98.03 (speaker accuracy)	 11.55 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	98.03 (speaker accuracy)	 11.55 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 8.73
EER: 8.47
Cllr (min/act): 0.303992 0.840626
linkability: 0.791232
```

</details>

### Oracle Wasserstein Procrustes
```sh
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --wp true --retrained-anon-xtractor true --oracle-f true
./run.sh --stage 1 --frontend-train "--pca --pca_n_dim 70" --wp true --retrained-anon-xtractor true --oracle-m true
```

<details>

<summary> results (merged):</summary>

```wiki
   == Data used to train rotation ==
     - xvect_libri_test_trials_f 
     - xvect_libri_test_trials_f_anon

Filtering by gender f
x_vector_l samples: 734
x_vector_u samples: 734
x_vector_l samples after filtering: 734
x_vector_u samples after filtering: 734
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (734, 70) with a total explained variance ratio on clear data: 0.98977935
Output shape after PCA: (734, 70) total explained variance ratio on target(anonymized) data: 0.97263455
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	99.32 (speaker accuracy)	 8.45 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Loading pca from: exp/enroll_train_wp
Top   1:	99.32 (speaker accuracy)	 8.45 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 13.09
EER: 13.00
Cllr (min/act): 0.463425 0.866338
linkability: 0.694130


   == Data used to train rotation ==
     - xvect_libri_test_trials_m 
     - xvect_libri_test_trials_m_anon

Filtering by gender m
x_vector_l samples: 762
x_vector_u samples: 762
x_vector_l samples after filtering: 762
x_vector_u samples after filtering: 762
== Training rotation ==
Computing PCA, 70 dimensions
Output shape after PCA: (762, 70) with a total explained variance ratio on clear data: 0.9897554
Output shape after PCA: (762, 70) total explained variance ratio on target(anonymized) data: 0.9801236
Wasserstein Procrustes rotation estimation
Compute done, rotation shape : (70, 70)
Top   1:	98.82 (speaker accuracy)	 7.35 (segment accuracy)
Done
== TEST rotation irreversibility ==
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Loading pca from: exp/enroll_train_wp
Top   1:	98.82 (speaker accuracy)	 7.35 (segment accuracy)
== TEST likability between Anonymized and Orignal speech ==
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech) 
Loading pca from: exp/enroll_train_wp
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
Loading pca from: exp/enroll_train_wp
ROC_EER: 10.49
EER: 10.11
Cllr (min/act): 0.342906 0.849621
linkability: 0.746411
```

</details>


#### Create this file by with vim

Put your cursor on the shell command and type @q (@b for the second line if any).  
The result block will be updated

The `q` macro (be a ; : remapper!):
```viml
let @q = '^"ay$/wikijV/``kdk;read ! a --verbose "--noverbose" | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"'
let @b = '^"ay$/wikij/``koo;read ! a --verbose "--noverbose" | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"'
```
