# Between-participant variability in matched-load HIFT

Reproducibility package for a Bayesian secondary analysis of between-participant variability in perceived exertion and physiological responses to matched-load high-intensity functional training.

## Author

Simon Salaj

## Repository contents

```text
analysis/   Python analysis scripts and Stan models
data/       Analysis-ready public data
results/    Machine-readable numerical results and model checks
```

Version 1.1.0 adds the numerical analyses used for the manuscript revision. The relaxed-prior RPE and lactate specifications are the revised primary models; calibrated specifications are retained as sensitivity analyses.

## Reproduce the analysis

Python 3.12 and CmdStan 2.36.0 were used for the archived results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m cmdstanpy.install_cmdstan --version 2.36.0

python analysis/01_original_models.py --stage all
python analysis/02_original_prior_predictive.py
python analysis/03_primary_analysis.py
python analysis/04_supportive_analysis.py
python analysis/05_leave_one_out.py
python analysis/06_revision_sensitivity.py --analysis all
```

The full run is computationally intensive: the leave-one-participant-out step repeatedly refits both models, and the revision-sensitivity step fits both likelihood- and prior-power grids. Use `python analysis/01_original_models.py --stage core` to regenerate only the original models required by the later scripts. Set `HIFT_CMDSTAN` if CmdStan is installed outside the default CmdStanPy location.

Generated posterior chains are written below `results/` but are excluded from version control. The final command adds likelihood power-scaling, prior power-scaling, and the observed-scale ordinal RPE VPC. The concise JSON and CSV results needed to audit the manuscript are retained.

## Data provenance

The analysis-ready files were derived from:

Oliver-López A, García-Valverde A, Sabido R. *DataBase_Modalities_HIFT*. Figshare. https://doi.org/10.6084/m9.figshare.25858906.v1

The source dataset is licensed under CC BY 4.0. The source workbook MD5 was `6cf222762295ce5339f041c4f0999999`. The transformation audit is provided in `data/data_audit.json`.

## Results retained in the archive

- calibrated and relaxed-prior RPE and lactate summaries;
- the original-model sensitivity grid and prior-predictive summary;
- prior- and posterior-predictive numerical checks;
- individual posterior expectations;
- prior-sensitivity comparisons;
- supportive-outcome estimand checks; and
- all leave-one-participant-out VPC refits and their diagnostics;
- likelihood and prior power-scaling summaries;
- relaxed-primary participant predictions and posterior-predictive checks; and
- the observed-scale analogue of the ordinal RPE VPC.

The source-analysis summaries are under `results/original/`. Revised primary results, predictions, posterior-predictive checks, and calibrated sensitivity results are under `results/final/`. Additional power-scaling and observed-scale VPC outputs are under `results/revision/`.

## Citation

For a GitHub-only citation use the following; after Zenodo archives version 1.1.0, prefer its version DOI:

> Salaj S. Between-participant variability in matched-load HIFT: Reproducibility package. Version 1.1.0. 2026. https://github.com/simonator420/hift-between-participant-variability

## License

Analysis code is released under the [MIT License](LICENSE). The included data remain subject to the source dataset's CC BY 4.0 license and attribution requirement.
