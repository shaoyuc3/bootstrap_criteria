# bootstrap_criteria

Code for the participant-bootstrap significance criteria used in the OpenFace AU within-vs-between association figures.

## Method

The bootstrap unit is the participant/interview, not the flattened segment row. For each bootstrap draw, participants are sampled with replacement.

- **Within-participant association:** AU features and target scores are centered within each participant, then the within-participant Pearson correlation is recomputed from participant-level sufficient statistics.
- **Between-participant association:** AU features and target scores are averaged within each participant, then the between-participant Pearson correlation is recomputed across sampled participant means.
- **Significance criterion:** an AU is marked significant on an axis when the participant-bootstrap 95% confidence interval for that axis excludes zero.

This addresses the degrees-of-freedom concern by resampling whole participants instead of treating all transcript segments as independent observations.

## Files

- `src/bootstrap_criteria/participant_bootstrap.py`: reusable implementation of the within/between participant-bootstrap criteria.
- `scripts/run_bootstrap_significance.py`: command-line runner for a segment-level CSV.

## Python example

```python
import pandas as pd
from bootstrap_criteria import participant_bootstrap_inference

df = pd.read_csv("segment_level_au_scores.csv")
feature_cols = ["AU01_r", "AU02_r", "AU04_r"]

results = participant_bootstrap_inference(
    df,
    feature_cols,
    participant_col="pid",
    score_col="score",
    n_boot=10000,
    seed=20260828,
)
print(results)
```

## CLI example

```bash
python scripts/run_bootstrap_significance.py \
  --input segment_level_au_scores.csv \
  --output bootstrap_significance.csv \
  --participant-col pid \
  --score-col score \
  --feature-cols AU01_r AU02_r AU04_r AU05_r \
  --n-boot 10000
```

The input dataframe should contain one row per scored segment, a participant ID column, a target-score column, and one or more AU feature columns.
