#!/usr/bin/env python3
"""Representative prior-predictive checks for the primary RPE and lactate models."""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]


def q(x: np.ndarray) -> dict[str, float]:
    a = np.quantile(x, [0.025, 0.5, 0.975])
    return {"q025": float(a[0]), "median": float(a[1]), "q975": float(a[2])}


def common(d: pd.DataFrame):
    ids = sorted(d.id.astype(int).unique().tolist())
    lookup = {pid: i + 1 for i, pid in enumerate(ids)}
    participant = d.id.map(lookup).to_numpy(int)
    ft = (d.workout == "FT").astype(float).to_numpy()
    emom = (d.workout == "EMOM").astype(float).to_numpy()
    work = d.work_repetitions_z.to_numpy(float)
    return ids, participant, ft, emom, work


def main() -> None:
    if os.environ.get("HIFT_CMDSTAN"):
        set_cmdstan_path(os.environ["HIFT_CMDSTAN"])
    else:
        cmdstan_path()
    d = pd.read_csv(ROOT / "data" / "hift_long_complete_case.csv")
    ids, participant, ft, emom, work = common(d)
    ordinal = CmdStanModel(stan_file=str(ROOT / "analysis" / "models" / "ordinal_random_intercept.stan"))
    levels = list(range(int(d.rpe.min()), int(d.rpe.max()) + 1))
    lookup = {level: i + 1 for i, level in enumerate(levels)}
    od = {
        "N": len(d), "J": len(ids), "K": len(levels), "P": 3,
        "y": d.rpe.astype(int).map(lookup).to_numpy(int),
        "participant": participant, "X": np.column_stack([ft, emom, work]),
        "prior_scale": 1.0, "prior_only": 1,
    }
    (ROOT / "results" / "fits" / "prior_rpe").mkdir(parents=True, exist_ok=True)
    ofit = ordinal.sample(
        data=od, chains=4, parallel_chains=4, iter_warmup=1000,
        iter_sampling=1000, seed=20260924, adapt_delta=0.95,
        inits=0, show_progress=False,
        output_dir=str(ROOT / "results" / "fits" / "prior_rpe"),
    )
    oy = ofit.stan_variable("y_rep").astype(int)

    gaussian = CmdStanModel(stan_file=str(ROOT / "analysis" / "models" / "gaussian_random_intercept.stan"))
    log_post = np.log(d.lactate_post)
    y_mean, y_sd = float(log_post.mean()), float(log_post.std(ddof=1))
    y = ((log_post - y_mean) / y_sd).to_numpy()
    log_pre = np.log(d.lactate_pre)
    baseline = ((log_pre - log_pre.mean()) / log_pre.std(ddof=1)).to_numpy()
    gd = {
        "N": len(d), "J": len(ids), "P": 5, "y": y,
        "participant": participant,
        "X": np.column_stack([np.ones(len(d)), ft, emom, work, baseline]),
        "prior_scale": 1.0, "prior_only": 1,
    }
    (ROOT / "results" / "fits" / "prior_lactate").mkdir(parents=True, exist_ok=True)
    gfit = gaussian.sample(
        data=gd, chains=4, parallel_chains=4, iter_warmup=1000,
        iter_sampling=1000, seed=20260924, adapt_delta=0.95,
        inits=0, show_progress=False,
        output_dir=str(ROOT / "results" / "fits" / "prior_lactate"),
    )
    gy = gfit.stan_variable("y_rep")
    log_original = gy * y_sd + y_mean
    result = {
        "rpe": {
            "levels": levels,
            "replicated_category_count": {
                str(level): q((oy == i + 1).sum(axis=1)) for i, level in enumerate(levels)
            },
            "replicated_mean_category": q(np.array(levels)[oy - 1].mean(axis=1)),
        },
        "lactate": {
            "scale": "original mmol/L after exponentiating log predictions",
            "replicated_geometric_mean": q(np.exp(log_original.mean(axis=1))),
            "replicated_min": q(np.exp(log_original).min(axis=1)),
            "replicated_max": q(np.exp(log_original).max(axis=1)),
            "note": "Heavy-tailed half-Student-t variance priors intentionally allow rare extreme datasets.",
        },
    }
    path = ROOT / "results" / "original" / "checks" / "prior_predictive_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
