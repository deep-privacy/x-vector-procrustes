x-vector-procrustes
===================

```sh
$ ./run.sh
Reproduce VoicePrivacy EER results with cosine scoring
**ASV (original): test_trials_f original <=> test_enrolls - original**
EER: 10.33%
Threshold: 0.43
**ASV (original): test_trials_m original <=> test_enrolls - original**
EER: 2.85%
Threshold: 0.52
**ASV (original): test_trials_f anonymized <=> test_enrolls - original**
EER: 48.99%
Threshold: 0.25
**ASV (original): test_trials_m anonymized <=> test_enrolls - original**
EER: 42.64%
Threshold: 0.12
**ASV (original): test_trials_f anonymized <=> test_enrolls - anonymized**
EER: 29.41%
Threshold: 0.86
**ASV (original): test_trials_m anonymized <=> test_enrolls - anonymized**
EER: 29.15%
Threshold: 0.88

Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV (anon): test_trials_f anonymized <=> test_enrolls - anonymized**
EER: 17.12%
Threshold: 0.85
**ASV (anon): test_trials_m anonymized <=> test_enrolls - anonymized**
EER: 14.08%
Threshold: 0.89

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
   DATA prep:
     - xvect_libri_test_enrolls
     - xvect_libri_test_enrolls_anon
 == Data used to train procrustes uv ==
== Training procrustes UV ==
Data Loaded :      (438, 512) (438, 512) (438,) (438,)
Frontend applied : (438, 512) (438, 512) (438,) (438,)
Compute done, rotation shape : (512, 512)
Top   1:        97.95    30.59
Done
== TEST procrustes UV ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Data Loaded :      (438, 512) (438, 512) (438,) (438,)
Frontend applied : (438, 512) (438, 512) (438,) (438,)
Top   1:        97.95    30.59
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Data Loaded :      (734, 512) (734, 512) (734,) (734,)
Frontend applied : (734, 512) (734, 512) (734,) (734,)
Top   1:        58.72    3.27
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Data Loaded :      (762, 512) (762, 512) (762,) (762,)
Frontend applied : (762, 512) (762, 512) (762,) (762,)
Top   1:        59.19    2.89
Perform likability between Anonymized and Orignal speech
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized <=> test_enrolls - original**
EER: 48.53%
Threshold: -0.05
**ASV: test_trials_m anonymized <=> test_enrolls - original**
EER: 49.75%
Threshold: -0.07
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 28.99%
Threshold: 0.51
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 23.65%
Threshold: 0.50
```

```sh
./run.sh --retrained_anon_xtractor false
Reproduce VoicePrivacy EER results with cosine scoring
**ASV (original): test_trials_f original <=> test_enrolls - original**
EER: 10.33%
Threshold: 0.43
**ASV (original): test_trials_m original <=> test_enrolls - original**
EER: 2.85%
Threshold: 0.52
**ASV (original): test_trials_f anonymized <=> test_enrolls - original**
EER: 48.99%
Threshold: 0.25
**ASV (original): test_trials_m anonymized <=> test_enrolls - original**
EER: 42.64%
Threshold: 0.12
**ASV (original): test_trials_f anonymized <=> test_enrolls - anonymized**
EER: 29.41%
Threshold: 0.86
**ASV (original): test_trials_m anonymized <=> test_enrolls - anonymized**
EER: 29.15%
Threshold: 0.88

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
   DATA prep:
     - xvect_libri_test_enrolls
     - xvect_libri_test_enrolls_anon
 == Data used to train procrustes uv ==
== Training procrustes UV ==
Data Loaded :      (438, 512) (438, 512) (438,) (438,)
Frontend applied : (438, 512) (438, 512) (438,) (438,)
Compute done, rotation shape : (512, 512)
Top   1:        96.12    34.47
Done
== TEST procrustes UV ==
**Accuracy enrolls anonymized => procrustes => enrolls - original**
Data Loaded :      (438, 512) (438, 512) (438,) (438,)
Frontend applied : (438, 512) (438, 512) (438,) (438,)
Top   1:        96.12    34.47
**Accuracy trials_f anonymized => procrustes => trials_f - original**
Data Loaded :      (734, 512) (734, 512) (734,) (734,)
Frontend applied : (734, 512) (734, 512) (734,) (734,)
Top   1:        25.48    1.36
**Accuracy trials_m anonymized => procrustes => trials_m - original**
Data Loaded :      (762, 512) (762, 512) (762,) (762,)
Frontend applied : (762, 512) (762, 512) (762,) (762,)
Top   1:        36.61    2.36
Perform likability between Anonymized and Orignal speech
  Anonymized x-vector -> (extracted by a x-vector trained on anonymized speech)
  Original x-vector -> (extracted by a x-vector trained on anonymized speech)
**ASV: test_trials_f anonymized <=> test_enrolls - original**
EER: 48.99%
Threshold: 0.25
**ASV: test_trials_m anonymized <=> test_enrolls - original**
EER: 42.64%
Threshold: 0.12
**ASV: test_trials_f anonymized => procrustes <=> test_enrolls - original**
EER: 41.94%
Threshold: 0.46
**ASV: test_trials_m anonymized => procrustes <=> test_enrolls - original**
EER: 30.55%
Threshold: 0.47

```
