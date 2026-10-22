# Dataset card: PTB-XL subset used here

* **Source.** PTB-XL v1.0.3, https://physionet.org/content/ptb-xl/1.0.3/ (fetched from PhysioNet's open-data AWS
  mirror and checked against PhysioNet's SHA256SUMS.txt). Licence CC BY 4.0. Cite Wagner et al., *Sci. Data* 7,
  154 (2020).
* **Contents.** 21,799 ten-second 12-lead ECGs from 18,869 patients, recorded 1989–1996 with Schiller devices, with
  SCP-ECG statements annotated by cardiologists. This project uses the 100 Hz versions.
* **Labels.** Diagnostic superclasses come from `scp_statements.csv`. NORM (label 0) means the superclass set is
  exactly {NORM}; MI (label 1) means it is exactly {MI}.

| | records |
|---|---|
| NORM | 9,069 |
| MI | 2,532 |
| excluded, other superclasses | 9,786 |
| excluded, no diagnostic superclass | 411 |
| excluded, both NORM and MI | 1 |

| split (folds) | records | patients | MI share |
|---|---|---|---|
| train (1–8) | 9,286 | 8,522 | 22.0% |
| validation (9) | 1,147 | 1,073 | 20.3% |
| test (10) | 1,168 | 1,057 | 21.9% |

Patients are disjoint across splits (checked in code). No record had non-finite values.

* **Preprocessing.** 0.5–40 Hz 4th-order zero-phase Butterworth band-pass, divide by 1 mV, clip to ±5 mV,
  float16 cache. No statistics are fitted to the data.
* **Summary statistics** use all 21,799 records. Eleven records whose heart rate could not be measured are dropped
  from the heart-rate queries by a per-record rule.
* **Limitations.** The data comes from one German centre and one era of devices; ages above 89 are coded as 300 in
  the source and capped at 90 here. Excluding mixed-label records makes the task easier than clinical practice.
  Nothing here is a clinical validation.
