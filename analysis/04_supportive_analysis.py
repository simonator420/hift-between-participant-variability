#!/usr/bin/env python3
"""Check whether supportive-outcome random intercepts represent comparable quantities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
MODEL_DIR = Path(os.environ.get("HIFT_MODEL_DIR", str(ROOT / "analysis" / "models")))
FIT_ROOT = Path(os.environ.get("HIFT_FIT_ROOT", str(OUT / "fits_supportive")))
SEED = 20260927


def q(x):
    z = np.quantile(x, [0.025, 0.5, 0.975])
    return {"q025": float(z[0]), "median": float(z[1]), "q975": float(z[2])}


def fit_one(model, d: pd.DataFrame, y_raw: pd.Series, name: str, include_predicted_hrmax: bool = False):
    y_mean, y_sd = float(y_raw.mean()), float(y_raw.std(ddof=1))
    y = ((y_raw - y_mean) / y_sd).to_numpy()
    ids = sorted(d.id.astype(int).unique().tolist()); lookup = {x: i + 1 for i, x in enumerate(ids)}
    cols = [
        np.ones(len(d)), (d.workout == "FT").astype(float).to_numpy(),
        (d.workout == "EMOM").astype(float).to_numpy(), d.work_repetitions_z.to_numpy(float),
    ]
    predictors = ["intercept", "workout_FT", "workout_EMOM", "work_z"]
    if include_predicted_hrmax:
        z = (d.predicted_hr_max - d.predicted_hr_max.mean()) / d.predicted_hr_max.std(ddof=1)
        cols.append(z.to_numpy()); predictors.append("predicted_hrmax_z")
    X = np.column_stack(cols)
    path = FIT_ROOT / name; path.mkdir(parents=True, exist_ok=True)
    fit = model.sample(
        data={"N": len(d), "J": len(ids), "P": X.shape[1], "y": y,
              "participant": d.id.map(lookup).to_numpy(int), "X": X,
              "prior_scale": 1.0, "prior_only": 0},
        chains=4, parallel_chains=4, iter_warmup=2000, iter_sampling=2000,
        seed=SEED, adapt_delta=0.99, max_treedepth=15, inits=0,
        output_dir=str(path), show_progress=False,
    )
    s = fit.summary(); mv = fit.method_variables(); beta = fit.stan_variable("beta")
    return {
        "model": name, "response_mean": y_mean, "response_sd": y_sd,
        "sigma_person_standardized": q(fit.stan_variable("sigma_person")),
        "sigma_person_original_units": q(fit.stan_variable("sigma_person") * y_sd),
        "vpc": q(fit.stan_variable("vpc")),
        "coefficients": {p: q(beta[:, i]) for i, p in enumerate(predictors)},
        "diagnostics": {
            "max_rhat": float(s.R_hat.max()), "min_ess_bulk": float(s.ESS_bulk.min()),
            "min_ess_tail": float(s.ESS_tail.min()),
            "divergences": int(np.asarray(mv["divergent__"]).sum()),
        },
    }


def original_vpc(name: str):
    d = json.loads((ROOT / "results" / "original" / "summaries" / f"{name}.json").read_text())
    return d["vpc"]


def main():
    (OUT / "summaries").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    if os.environ.get("HIFT_CMDSTAN"):
        set_cmdstan_path(os.environ["HIFT_CMDSTAN"])
    else:
        cmdstan_path()
    d = pd.read_csv(ROOT / "data" / "hift_long_complete_case.csv")
    model = CmdStanModel(stan_file=str(MODEL_DIR / "gaussian_random_intercept.stan"))
    outcomes = {
        "cmj_pre_minus_post": (d.cmj_pre - d.cmj_post, False),
        "hrv_pre_minus_post": (d.hrv_pre - d.hrv_post, False),
        "hr_mean_percent_predicted_max": (100 * d.hr_mean / d.predicted_hr_max, False),
        "hr_max_percent_predicted_max": (100 * d.hr_max / d.predicted_hr_max, False),
        "hr_mean_adjusted_predicted_max": (d.hr_mean, True),
        "hr_max_adjusted_predicted_max": (d.hr_max, True),
    }
    results = {name: fit_one(model, d, y, name, cov) for name, (y, cov) in outcomes.items()}
    (OUT / "summaries" / "supportive_model_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = [
        {"outcome": "CMJ", "estimand": "Post conditional on baseline", **original_vpc("cmj_gaussian_complete")},
        {"outcome": "CMJ", "estimand": "Pre-minus-post fatigue change", **results["cmj_pre_minus_post"]["vpc"]},
        {"outcome": "HRV", "estimand": "Post conditional on baseline", **original_vpc("hrv_gaussian_complete")},
        {"outcome": "HRV", "estimand": "Pre-minus-post suppression", **results["hrv_pre_minus_post"]["vpc"]},
        {"outcome": "Mean HR", "estimand": "Absolute bpm", **original_vpc("hr_mean_gaussian_complete")},
        {"outcome": "Mean HR", "estimand": "% predicted HRmax", **results["hr_mean_percent_predicted_max"]["vpc"]},
        {"outcome": "Mean HR", "estimand": "Absolute bpm adjusted for predicted HRmax", **results["hr_mean_adjusted_predicted_max"]["vpc"]},
        {"outcome": "Maximum HR", "estimand": "Absolute bpm", **original_vpc("hr_max_gaussian_complete")},
        {"outcome": "Maximum HR", "estimand": "% predicted HRmax", **results["hr_max_percent_predicted_max"]["vpc"]},
        {"outcome": "Maximum HR", "estimand": "Absolute bpm adjusted for predicted HRmax", **results["hr_max_adjusted_predicted_max"]["vpc"]},
    ]
    pd.DataFrame(rows).to_csv(OUT / "tables" / "supportive_estimand_comparison.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
