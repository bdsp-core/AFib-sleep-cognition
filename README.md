# AFib-sleep-cognition

**Public.** Code for our analysis of atrial fibrillation, overnight sleep
EEG, and downstream cognitive outcomes in the MrOS cohort.

## Layout

```
MrOS-analysis/
  extract_features_parallel.py   parallel feature extraction from MrOS PSGs
  dataset.xlsx                   curated subject-level table
  bandpower_features.csv         spectral features per subject
  AF_as_exposure*.xlsx           AF-as-exposure analysis tables
  AF_as_exposure_potential_outcomes*.pickle   potential-outcome estimates
  figures/                       figure-generation outputs
```

## Status

Paper-accompanying analysis code. Public; carries the standard noncommercial
LICENSE.
