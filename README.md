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

The repository intentionally excludes manuscripts, figures, compiled Stan files, posterior chain files, temporary files, and local software environments.

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
```

The full run is computationally intensive because the final step refits both primary models after omitting each participant. Use `python analysis/01_original_models.py --stage core` to regenerate only the original models required by the later scripts. Set `HIFT_CMDSTAN` if CmdStan is installed outside the default CmdStanPy location.

Generated posterior chains are written below `results/` but are excluded from version control. The concise JSON and CSV results needed to audit the manuscript are retained.

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
- all leave-one-participant-out VPC refits and their diagnostics.

## Citation

Until a Zenodo DOI is assigned, cite this repository as:

> Salaj S. Between-participant variability in matched-load HIFT: Reproducibility package. Version 1.0.0. 2026. https://github.com/simonator420/hift-between-participant-variability

## License

Analysis code is released under the [MIT License](LICENSE). The included data remain subject to the source dataset's CC BY 4.0 license and attribution requirement.
