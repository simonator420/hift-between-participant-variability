#!/usr/bin/env python3
"""Run the prespecified HIFT Bayesian multilevel models and sensitivities."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "original"
MODELS = ROOT / "analysis" / "models"
SEED = 20260924
WORKOUTS = ["AMRAP", "FT", "EMOM"]


@dataclass(frozen=True)
class Outcome:
    name: str
    response: str
    baseline: str | None = None
    log_response: bool = False
    log_baseline: bool = False
    unit: str = ""


OUTCOMES = {
    "lactate": Outcome("lactate", "lactate_post", "lactate_pre", True, True, "log mmol/L"),
    "cmj": Outcome("cmj", "cmj_post", "cmj_pre", False, False, "cm"),
    "hrv": Outcome("hrv", "hrv_post", "hrv_pre", False, False, "source units"),
    "hr_mean": Outcome("hr_mean", "hr_mean", unit="beats/min"),
    "hr_max": Outcome("hr_max", "hr_max", unit="beats/min"),
}


def zscore(x: pd.Series) -> tuple[np.ndarray, float, float]:
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardize a constant or missing vector")
    return ((x - mean) / sd).to_numpy(), mean, sd


def design_matrix(
    df: pd.DataFrame,
    *,
    include_intercept: bool,
    include_work: bool = True,
    baseline: np.ndarray | None = None,
    extra: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    if include_intercept:
        cols.append(np.ones(len(df)))
        names.append("intercept")
    cols.extend(
        [
            (df["workout"] == "FT").astype(float).to_numpy(),
            (df["workout"] == "EMOM").astype(float).to_numpy(),
        ]
    )
    names.extend(["workout_FT", "workout_EMOM"])
    if include_work:
        cols.append(df["work_repetitions_z"].to_numpy(float))
        names.append("work_repetitions_z")
    if baseline is not None:
        cols.append(baseline)
        names.append("baseline_z")
    if extra is not None:
        cols.append(df[extra].to_numpy(float))
        names.append(extra)
    return np.column_stack(cols), names


def participant_index(df: pd.DataFrame) -> tuple[np.ndarray, int, list[int]]:
    ids = sorted(df["id"].astype(int).unique().tolist())
    lookup = {pid: i + 1 for i, pid in enumerate(ids)}
    return df["id"].map(lookup).to_numpy(int), len(ids), ids


def q(draws: np.ndarray) -> dict[str, float]:
    vals = np.quantile(draws, [0.025, 0.5, 0.975])
    return {"q025": float(vals[0]), "median": float(vals[1]), "q975": float(vals[2])}


def diagnostics(fit) -> dict:
    summary = fit.summary()
    mv = fit.method_variables()
    return {
        "max_rhat": float(summary["R_hat"].replace([np.inf, -np.inf], np.nan).max()),
        "min_ess_bulk": float(summary["ESS_bulk"].min()),
        "min_ess_tail": float(summary["ESS_tail"].min()),
        "divergences": int(np.asarray(mv["divergent__"]).sum()),
        "max_treedepth_observed": int(np.asarray(mv["treedepth__"]).max()),
    }


def sample_model(model: CmdStanModel, data: dict, name: str, quick: bool):
    output_dir = OUTPUT / "fits" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return model.sample(
        data=data,
        chains=4,
        parallel_chains=4,
        iter_warmup=300 if quick else 2000,
        iter_sampling=300 if quick else 2000,
        seed=SEED,
        adapt_delta=0.95 if quick else 0.99,
        max_treedepth=15,
        inits=0,
        output_dir=str(output_dir),
        show_progress=True,
        refresh=100,
    )


def save_common(
    fit,
    name: str,
    predictor_names: list[str],
    metadata: dict,
    y_observed: np.ndarray,
    y_sd: float | None,
    ordinal_levels: list[int] | None = None,
) -> dict:
    result_dir = OUTPUT / "summaries"
    result_dir.mkdir(parents=True, exist_ok=True)
    fit.summary().to_csv(result_dir / f"{name}_parameters.csv")
    diag = diagnostics(fit)
    sigma = fit.stan_variable("sigma_person")
    vpc = fit.stan_variable("vpc")
    row = {
        "model": name,
        "sigma_person_model_scale": q(sigma),
        "vpc": q(vpc),
        "diagnostics": diag,
        **metadata,
    }
    if y_sd is not None:
        row["sigma_person_outcome_scale"] = q(sigma * y_sd)
    beta = fit.stan_variable("beta")
    row["coefficients"] = {
        predictor_names[i]: q(beta[:, i]) for i in range(len(predictor_names))
    }
    y_rep = fit.stan_variable("y_rep")
    if ordinal_levels is not None:
        row["posterior_predictive"] = {
            str(level): {
                "observed_n": int((y_observed == i + 1).sum()),
                "replicated_n": q((y_rep == i + 1).sum(axis=1)),
            }
            for i, level in enumerate(ordinal_levels)
        }
    else:
        row["posterior_predictive"] = {
            "observed_mean": float(np.mean(y_observed)),
            "replicated_mean": q(y_rep.mean(axis=1)),
            "observed_sd": float(np.std(y_observed, ddof=1)),
            "replicated_sd": q(y_rep.std(axis=1, ddof=1)),
            "observed_min": float(np.min(y_observed)),
            "replicated_min": q(y_rep.min(axis=1)),
            "observed_max": float(np.max(y_observed)),
            "replicated_max": q(y_rep.max(axis=1)),
        }
    (result_dir / f"{name}.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return row


def fit_ordinal(
    df: pd.DataFrame,
    model: CmdStanModel,
    name: str,
    *,
    include_work: bool = True,
    extra: str | None = None,
    prior_scale: float = 1.0,
    quick: bool = False,
) -> dict:
    d = df.dropna(subset=["rpe"]).copy()
    observed_levels = sorted(d["rpe"].astype(int).unique().tolist())
    # Preserve every integer CR-10 category between the observed endpoints.
    levels = list(range(min(observed_levels), max(observed_levels) + 1))
    lookup = {level: i + 1 for i, level in enumerate(levels)}
    y = d["rpe"].astype(int).map(lookup).to_numpy(int)
    X, names = design_matrix(
        d, include_intercept=False, include_work=include_work, extra=extra
    )
    pid, J, ids = participant_index(d)
    stan_data = {
        "N": len(d), "J": J, "K": len(levels), "P": X.shape[1],
        "y": y, "participant": pid, "X": X, "prior_scale": prior_scale,
        "prior_only": 0,
    }
    fit = sample_model(model, stan_data, name, quick)
    return save_common(
        fit, name, names,
        {
            "outcome": "RPE", "family": "cumulative_logit",
            "n_observations": len(d), "n_participants": J,
            "participant_ids": ids, "rpe_levels": levels,
            "prior_scale": prior_scale,
        },
        y, None, levels,
    )


def fit_continuous(
    df: pd.DataFrame,
    outcome: Outcome,
    model: CmdStanModel,
    family: str,
    name: str,
    *,
    include_work: bool = True,
    extra: str | None = None,
    prior_scale: float = 1.0,
    quick: bool = False,
) -> dict:
    required = [outcome.response]
    if outcome.baseline:
        required.append(outcome.baseline)
    d = df.dropna(subset=required).copy()
    y_raw = d[outcome.response].astype(float)
    if outcome.log_response:
        y_raw = np.log(y_raw)
    y, y_mean, y_sd = zscore(y_raw)
    baseline = None
    baseline_meta = None
    if outcome.baseline:
        b = d[outcome.baseline].astype(float)
        if outcome.log_baseline:
            b = np.log(b)
        baseline, b_mean, b_sd = zscore(b)
        baseline_meta = {"mean": b_mean, "sd": b_sd, "logged": outcome.log_baseline}
    X, names = design_matrix(
        d, include_intercept=True, include_work=include_work,
        baseline=baseline, extra=extra,
    )
    pid, J, ids = participant_index(d)
    stan_data = {
        "N": len(d), "J": J, "P": X.shape[1], "y": y,
        "participant": pid, "X": X, "prior_scale": prior_scale,
        "prior_only": 0,
    }
    fit = sample_model(model, stan_data, name, quick)
    return save_common(
        fit, name, names,
        {
            "outcome": outcome.name, "family": family,
            "n_observations": len(d), "n_participants": J,
            "participant_ids": ids, "prior_scale": prior_scale,
            "response_transform": "natural_log" if outcome.log_response else "identity",
            "response_center": y_mean, "response_scale": y_sd,
            "response_unit": outcome.unit, "baseline_transform": baseline_meta,
        },
        y, y_sd,
    )


def model_plan(stage: str) -> Iterable[tuple[str, dict]]:
    # Each yielded item is (kind, arguments). The complete stage includes every
    # prespecified sensitivity plus the limited sex/strength checks.
    yield "ordinal", {"name": "rpe_ordinal_complete"}
    for outcome in OUTCOMES.values():
        yield "continuous", {
            "outcome": outcome, "family": "gaussian",
            "name": f"{outcome.name}_gaussian_complete",
        }
    if stage == "core":
        return

    yield "continuous", {
        "outcome": Outcome("rpe", "rpe", unit="CR-10 points"),
        "family": "gaussian", "name": "rpe_gaussian_complete",
    }
    yield "ordinal", {"name": "rpe_ordinal_all_available", "dataset": "all"}
    yield "ordinal", {"name": "rpe_ordinal_no_work", "include_work": False}
    yield "ordinal", {"name": "rpe_ordinal_experience", "extra": "experience_months_z"}
    yield "ordinal", {"name": "rpe_ordinal_wide_prior", "prior_scale": 2.0}
    yield "ordinal", {"name": "rpe_ordinal_sex", "extra": "sex_woman"}
    yield "ordinal", {"name": "rpe_ordinal_strength", "extra": "relative_push_press_z"}
    for outcome in OUTCOMES.values():
        yield "continuous", {
            "outcome": outcome, "family": "student_t",
            "name": f"{outcome.name}_student_t_complete",
        }
        yield "continuous", {
            "outcome": outcome, "family": "gaussian",
            "name": f"{outcome.name}_gaussian_all_available", "dataset": "all",
        }
        yield "continuous", {
            "outcome": outcome, "family": "gaussian",
            "name": f"{outcome.name}_gaussian_no_work", "include_work": False,
        }
        yield "continuous", {
            "outcome": outcome, "family": "gaussian",
            "name": f"{outcome.name}_gaussian_experience", "extra": "experience_months_z",
        }
        yield "continuous", {
            "outcome": outcome, "family": "gaussian",
            "name": f"{outcome.name}_gaussian_wide_prior", "prior_scale": 2.0,
        }
    for extra, label in [("sex_woman", "sex"), ("relative_push_press_z", "strength")]:
        yield "continuous", {
            "outcome": OUTCOMES["lactate"], "family": "gaussian",
            "name": f"lactate_gaussian_{label}", "extra": extra,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["core", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Short test run, not publication-ready")
    args = parser.parse_args()
    if os.environ.get("HIFT_CMDSTAN"):
        set_cmdstan_path(os.environ["HIFT_CMDSTAN"])
    else:
        cmdstan_path()
    complete = pd.read_csv(DATA / "hift_long_complete_case.csv")
    all_available = pd.read_csv(DATA / "hift_long_all_available.csv")

    ordinal_model = CmdStanModel(stan_file=str(MODELS / "ordinal_random_intercept.stan"))
    gaussian_model = CmdStanModel(stan_file=str(MODELS / "gaussian_random_intercept.stan"))
    student_model = CmdStanModel(stan_file=str(MODELS / "student_t_random_intercept.stan"))
    fitted: list[dict] = []
    for kind, spec in model_plan(args.stage):
        spec = dict(spec)
        dataset = spec.pop("dataset", "complete")
        df = all_available if dataset == "all" else complete
        print(f"\n=== {spec['name']} ===", flush=True)
        if kind == "ordinal":
            fitted.append(fit_ordinal(df, ordinal_model, quick=args.quick, **spec))
        else:
            family = spec.pop("family")
            model = gaussian_model if family == "gaussian" else student_model
            fitted.append(
                fit_continuous(df, model=model, family=family, quick=args.quick, **spec)
            )

    rows = []
    for item in fitted:
        rows.append(
            {
                "model": item["model"], "outcome": item["outcome"],
                "family": item["family"], "n": item["n_observations"],
                "participants": item["n_participants"],
                "sigma_median": item["sigma_person_model_scale"]["median"],
                "sigma_q025": item["sigma_person_model_scale"]["q025"],
                "sigma_q975": item["sigma_person_model_scale"]["q975"],
                "vpc_median": item["vpc"]["median"],
                "vpc_q025": item["vpc"]["q025"],
                "vpc_q975": item["vpc"]["q975"],
                **item["diagnostics"],
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT / "tables" / "all_model_summary.csv", index=False)


if __name__ == "__main__":
    main()
